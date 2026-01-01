import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# ================= [설정] =================
DATA_DIR = "/data/dataset/mlp"
INPUT_FILE = os.path.join(DATA_DIR, "vectors_student_aligned.pt")
TARGET_FILE = os.path.join(DATA_DIR, "vectors_teacher_aligned.pt")
SAVE_MODEL_PATH = os.path.join(DATA_DIR, "eva_projector_vqvae.pth")

BATCH_SIZE = 1024       # VQ-VAE는 배치가 너무 작으면 코드북 학습이 잘 안됨
EPOCHS = 100            # 충분한 학습 필요
LEARNING_RATE = 1e-3
COMMITMENT_COST = 0.25  # Encoder가 코드북에서 너무 멀어지지 않게 잡는 강도
NUM_EMBEDDINGS = 4096   # 코드북 단어 개수 (Vocabulary Size)
EMBEDDING_DIM = 1280    # 코드북 벡터 차원 (Teacher와 동일)
decay = 0.99            # EMA 감쇠율 (높을수록 코드북이 천천히 변함)
# =======================================

class VectorQuantizerEMA(nn.Module):
    """
    VQ-VAE의 핵심 모듈 (EMA 방식 적용)
    경사하강법 대신 이동평균(Moving Average)으로 코드북을 업데이트하여 학습이 훨씬 안정적임.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 코드북 (학습 파라미터가 아닌 Buffer로 등록하여 Optimizer가 건드리지 않게 함)
        self.register_buffer('embedding', torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.randn(num_embeddings, embedding_dim))
        
        self.decay = decay
        self.epsilon = epsilon

    def forward(self, inputs):
        # inputs: [Batch, Channel] -> [Batch, Embedding_Dim]
        # 거리 계산 (x^2 + e^2 - 2xe)
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding**2, dim=1)
                    - 2 * torch.matmul(inputs, self.embedding.t()))
            
        # 가장 가까운 코드 찾기 (Encoding)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # [B, 1]
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1) # One-hot encoding
        
        # Quantize (코드북 값으로 치환)
        quantized = torch.matmul(encodings, self.embedding) # [B, Dim]
        
        # 학습 모드일 때만 코드북 업데이트 (EMA)
        if self.training:
            # 1. 각 코드가 몇 번 선택됐는지 카운트 (Moving Average)
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * torch.sum(encodings, 0)
            
            # 0으로 나누기 방지 (Laplace Smoothing)
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon) / 
                (n + self.num_embeddings * self.epsilon) * n
            )
            
            # 2. 선택된 입력값들의 평균으로 코드북 위치 이동
            dw = torch.matmul(encodings.t(), inputs)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
            
            self.embedding = self.ema_w / self.ema_cluster_size.unsqueeze(1)
        
        # Loss 계산
        # Commitment Loss: Encoder 출력이 코드북 근처에 머물도록
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator (STE)
        # 역전파 시에는 quantized 함수를 건너뛰고 inputs로 기울기를 흘려보냄
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity (코드북 활용도 지표)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity

class VQVAEProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048):
        super().__init__()
        
        # 1. Encoder (Student -> Latent)
        # 차원을 1280(Teacher Size)으로 맞춰서 코드북에 넣음
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim) # Output_dim = 1280
        )
        
        # 2. Vector Quantizer (The Codebook)
        self.vq = VectorQuantizerEMA(NUM_EMBEDDINGS, output_dim, COMMITMENT_COST, decay)
        
        # 3. Decoder (Quantized -> Teacher)
        # 이미 차원이 1280이므로, 다듬는(Refine) 역할만 수행
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perplexity = self.vq(z)
        out = self.decoder(quantized)
        return out, vq_loss, perplexity

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"사용 가능한 GPU: {num_gpus}개")

    # 1. 데이터 로드
    print("벡터 파일 로딩 중...")
    try:
        inputs = torch.load(INPUT_FILE)
        targets = torch.load(TARGET_FILE)
    except FileNotFoundError:
        print("파일 없음.")
        return

    split_idx = int(inputs.shape[0] * 0.9)
    train_ds = TensorDataset(inputs[:split_idx], targets[:split_idx])
    val_ds = TensorDataset(inputs[split_idx:], targets[split_idx:])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # 2. 모델 설정
    model = VQVAEProjector(input_dim=inputs.shape[1], output_dim=targets.shape[1])
    
    if num_gpus > 1:
        print(f"DataParallel 활성화: {num_gpus} GPUs")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 3. 학습 루프
    print(f"\n학습 시작 (Epochs: {EPOCHS})")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_recon_loss = 0
        train_vq_loss = 0
        train_perplexity = 0
        
        for x, y_teacher in train_loader:
            x, y_teacher = x.to(device), y_teacher.to(device)
            
            # Forward
            y_pred, vq_loss, perplexity = model(x)
            
            # Loss = Reconstruction(MSE) + Commitment(VQ)
            # vq_loss는 DataParallel 통과 시 벡터로 나오므로 평균 취함
            vq_loss = vq_loss.mean()
            recon_loss = F.mse_loss(y_pred, y_teacher)
            loss = recon_loss + vq_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_recon_loss += recon_loss.item()
            train_vq_loss += vq_loss.item()
            train_perplexity += perplexity.mean().item() # Perplexity 모니터링 중요!

        scheduler.step()
        
        # 검증
        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for x, y_teacher in val_loader:
                x, y_teacher = x.to(device), y_teacher.to(device)
                y_pred, vq_loss, _ = model(x)
                
                loss = F.mse_loss(y_pred, y_teacher) + vq_loss.mean()
                val_total_loss += loss.item()

        avg_train_loss = train_recon_loss / len(train_loader)
        avg_perplexity = train_perplexity / len(train_loader)
        avg_val_loss = val_total_loss / len(val_loader)
        
        # [Perplexity 확인]
        # 4096개 코드 중 실제 몇 개나 쓰는지 보여줌.
        # 이 값이 너무 낮으면(예: 10 미만) 코드북 붕괴(Collapse) 일어난 것.
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 저장
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), SAVE_MODEL_PATH)
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {avg_train_loss:.4f} | PPX: {avg_perplexity:.1f} | Val: {avg_val_loss:.4f} (★ Saved)")
        else:
            if (epoch+1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {avg_train_loss:.4f} | PPX: {avg_perplexity:.1f} | Val: {avg_val_loss:.4f}")

if __name__ == "__main__":
    main()