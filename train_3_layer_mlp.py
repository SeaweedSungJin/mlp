import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# ================= 설정 =================
DATA_DIR = "/data/dataset/mlp"

INPUT_FILE = os.path.join(DATA_DIR, "vectors_student_aligned.pt")
TARGET_FILE = os.path.join(DATA_DIR, "vectors_teacher_aligned.pt")
SAVE_MODEL_PATH = os.path.join(DATA_DIR, "eva_projector_3_layer.pth")

BATCH_SIZE = 1024  
EPOCHS = 50        # Scheduler와 Early Saving이 있으므로 넉넉히 둬도 됨
LEARNING_RATE = 1e-3
# =======================================

# 2-layer 학습
# class EmbeddingProjector(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim=4096, dropout=0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim), # 학습 안정화
#             nn.GELU(),
#             nn.Dropout(dropout),      # 과적합 방지
#             nn.Linear(hidden_dim, output_dim)
#         )
        
#     def forward(self, x):
#         return self.net(x)

# 3-layer 학습
class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=4096, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            # === 1번째 층 (Expansion) ===
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),  # 1층 끝나고도 한번 끊어주는 게 좋음

            # === 2번째 층 (Deepening) [추가된 부분] ===
            nn.Linear(hidden_dim, hidden_dim), # 4096 -> 4096
            nn.LayerNorm(hidden_dim),          # [추천] 여기도 Norm을 해줘야 학습이 안 튐
            nn.GELU(),
            nn.Dropout(dropout),               # 여기도 과적합 방지

            # === 3번째 층 (Projection) ===
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def main():
    # 1. 벡터 파일 로드
    print("벡터 파일 로딩 중...")
    try:
        inputs = torch.load(INPUT_FILE)
        targets = torch.load(TARGET_FILE)
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
        return

    print(f"데이터 개수: {inputs.shape[0]}개")
    print(f"차원 변환: {inputs.shape[1]} -> {targets.shape[1]}")

    # 데이터셋 생성 (9:1)
    split_idx = int(inputs.shape[0] * 0.9)
    
    # [팁] GPU로 미리 올리지 않고 DataLoader에서 올리는 것이 일반적이나, 
    # 데이터가 VRAM에 다 들어간다면 여기서 .cuda() 해버리는게 학습 속도는 제일 빠릅니다.
    # DGX(A100/V100)라면 2GB 정도는 껌이니 미리 올리는 옵션도 고려 가능. 
    # 여기선 안전하게 기존 방식 유지.
    
    train_ds = TensorDataset(inputs[:split_idx], targets[:split_idx])
    val_ds = TensorDataset(inputs[split_idx:], targets[split_idx:])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. 모델 및 학습 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingProjector(inputs.shape[1], targets.shape[1]).to(device)
    
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # [추가] 스케줄러: 학습 후반부에 LR을 서서히 줄여줌 (성능 향상 핵심)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 3. 학습 루프
    print(f"\n학습 시작 (Device: {device})")
    
    best_val_loss = float('inf') # 최고 기록 저장용
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            pred = model(x)
            target_labels = torch.ones(x.shape[0]).to(device)
            loss = criterion(pred, y, target_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # 스케줄러 업데이트
        scheduler.step()
            
        # 검증 (Validation)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                target_labels = torch.ones(x.shape[0]).to(device)
                loss = criterion(pred, y, target_labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        current_lr = scheduler.get_last_lr()[0]

        # [추가] Best Model 저장 로직
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH) # 덮어쓰기
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.1e} (★ Best Saved)")
        else:
            if (epoch+1) % 5 == 0:
                 print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.1e}")

    print(f"\n최종 학습 완료! 가장 성능이 좋았던 모델(Val Loss: {best_val_loss:.4f})이 저장되었습니다.")

if __name__ == "__main__":
    main()