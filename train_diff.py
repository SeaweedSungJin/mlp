import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import math

# ================= [설정] =================
DATA_DIR = "/data/dataset/mlp"
INPUT_FILE = os.path.join(DATA_DIR, "vectors_student_aligned.pt")
TARGET_FILE = os.path.join(DATA_DIR, "vectors_teacher_aligned.pt")
SAVE_MODEL_PATH = os.path.join(DATA_DIR, "eva_projector_diffusion.pth")

BATCH_SIZE = 4096      # Diffusion은 배치가 클수록 좋습니다.
EPOCHS = 100           # 생성 모델은 학습이 좀 더 오래 걸립니다.
LEARNING_RATE = 1e-4   # Diffusion은 LR을 조금 낮추는 게 안정적입니다.
TIMESTEPS = 1000       # 노이즈 단계 (보통 1000 사용)
# =======================================

# [1] 시간 정보(t)를 벡터로 바꿔주는 모듈 (Sinusoidal Embedding)
class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# [2] 노이즈 예측 모델 (Denoising MLP)
class DiffusionProjector(nn.Module):
    def __init__(self, student_dim, teacher_dim, time_dim=256, hidden_dim=4096, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU()
        )
        
        # 입력: [Noisy_Teacher(1280) + Student_Condition(768) + Time(256)]
        input_total_dim = teacher_dim + student_dim + time_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, teacher_dim) # 출력: 예측된 노이즈 (Teacher 차원과 동일)
        )

    def forward(self, x, condition, t):
        # x: Noisy Teacher Vector
        # condition: Student Vector
        # t: Timestep
        
        t_emb = self.time_mlp(t)
        # 모든 정보를 이어 붙임 (Concatenate)
        x_input = torch.cat([x, condition, t_emb], dim=1)
        return self.net(x_input)

# [3] Diffusion 스케줄러 (노이즈 추가 및 스케줄 관리)
class VectorDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        # Beta Schedule (Linear)
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps)
        
        # 미리 계산해두는 값들 (register_buffer로 저장하면 GPU 자동 이동)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        """Forward Process: 원본 벡터에 노이즈 섞기"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, condition, t, noise=None):
        """Training Loss 계산"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # 1. 노이즈 섞인 Teacher 벡터 생성 (x_t)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # 2. 모델이 노이즈를 예측 (Student 벡터를 힌트로 사용)
        predicted_noise = self.model(x_noisy, condition, t)
        
        # 3. MSE Loss (정답 노이즈 vs 예측 노이즈)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        return loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"사용 가능한 GPU: {num_gpus}개 (할당된 장치만 보임)")

    # 1. 데이터 로드 (기존과 동일)
    print("벡터 파일 로딩 중...")
    try:
        inputs = torch.load(INPUT_FILE)   # Student (Condition)
        targets = torch.load(TARGET_FILE) # Teacher (Target)
    except FileNotFoundError:
        print("파일 없음.")
        return

    # 데이터셋 생성
    split_idx = int(inputs.shape[0] * 0.9)
    train_ds = TensorDataset(inputs[:split_idx], targets[:split_idx])
    val_ds = TensorDataset(inputs[split_idx:], targets[split_idx:])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # 2. 모델 초기화
    student_dim = inputs.shape[1]
    teacher_dim = targets.shape[1]
    
    base_model = DiffusionProjector(student_dim, teacher_dim)
    diffusion = VectorDiffusion(base_model, timesteps=TIMESTEPS)

    # [DataParallel 적용]
    # Diffusion 클래스 자체를 감싸는 게 편합니다.
    if num_gpus > 1:
        print(f"DataParallel 활성화: {num_gpus} GPUs")
        diffusion = nn.DataParallel(diffusion)
    
    diffusion = diffusion.to(device)
    
    optimizer = optim.AdamW(diffusion.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Cosine Annealing (Diffusion은 학습이 길어서 유용)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 3. 학습 루프
    print(f"\n학습 시작 (Epochs: {EPOCHS})")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        diffusion.train()
        train_loss = 0
        
        for student_vec, teacher_vec in train_loader:
            student_vec = student_vec.to(device)
            teacher_vec = teacher_vec.to(device)
            
            # 랜덤한 시간 t 샘플링
            batch = student_vec.shape[0]
            t = torch.randint(0, TIMESTEPS, (batch,), device=device).long()
            
            # DataParallel 내부에서 p_losses 호출 -> MSE Loss 반환
            if isinstance(diffusion, nn.DataParallel):
                loss = diffusion.module.p_losses(teacher_vec, student_vec, t)
            else:
                loss = diffusion.p_losses(teacher_vec, student_vec, t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        scheduler.step()
        
        # 검증 (MSE Loss 확인)
        diffusion.eval()
        val_loss = 0
        with torch.no_grad():
            for student_vec, teacher_vec in val_loader:
                student_vec = student_vec.to(device)
                teacher_vec = teacher_vec.to(device)
                
                t = torch.randint(0, TIMESTEPS, (student_vec.shape[0],), device=device).long()
                
                if isinstance(diffusion, nn.DataParallel):
                    loss = diffusion.module.p_losses(teacher_vec, student_vec, t)
                else:
                    loss = diffusion.p_losses(teacher_vec, student_vec, t)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 저장할 때는 껍데기 벗기고 저장
            model_to_save = diffusion.module.model if isinstance(diffusion, nn.DataParallel) else diffusion.model
            torch.save(model_to_save.state_dict(), SAVE_MODEL_PATH)
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} (★ Saved)")
        else:
            if (epoch+1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

if __name__ == "__main__":
    main()