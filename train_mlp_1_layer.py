import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# ================= 설정 =================
DATA_DIR = "/data/dataset/mlp"

INPUT_FILE = os.path.join(DATA_DIR, "vectors_student_aligned.pt")
TARGET_FILE = os.path.join(DATA_DIR, "vectors_teacher_aligned.pt")
SAVE_MODEL_PATH = os.path.join(DATA_DIR, "eva_projector_1_layer.pth") # 파일명 변경

BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATE = 1e-3
# =======================================

# [핵심 변경] 1-Layer (Linear Projection) 구조
class LinearProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 은닉층 없이 바로 매핑 (768 -> 1280)
        # 비선형성(GELU)이나 Dropout도 제거하여 순수한 선형 성능 측정
        self.net = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.net(x)

def main():
    print("벡터 파일 로딩 중...")
    try:
        inputs = torch.load(INPUT_FILE)
        targets = torch.load(TARGET_FILE)
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
        return

    split_idx = int(inputs.shape[0] * 0.9)
    train_ds = TensorDataset(inputs[:split_idx], targets[:split_idx])
    val_ds = TensorDataset(inputs[split_idx:], targets[split_idx:])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 모델 설정 (LinearProjector 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearProjector(inputs.shape[1], targets.shape[1]).to(device)
    
    # Baseline이므로 가장 기본적인 Cosine Loss 사용
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n학습 시작 (1-Layer Linear) - Device: {device}")
    best_val_loss = float('inf')
    
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
            
        scheduler.step()
            
        # 검증
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} (★ Best Saved)")
        else:
            if (epoch+1) % 5 == 0:
                 print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    print(f"최종 완료. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()