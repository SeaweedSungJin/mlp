from transformers import CLIPModel, CLIPConfig

model_id = "openai/clip-vit-large-patch14-336"

# 1. 설정(Config)만 가볍게 로드
config = CLIPConfig.from_pretrained(model_id)

# 2. 모델 전체 로드 (파라미터 수 계산용)
model = CLIPModel.from_pretrained(model_id)

print(f"=== 모델: {model_id} ===")
print(f"1. 총 파라미터 수: {model.num_parameters() / 1_000_000:.2f} Million")
print(f"2. Vision 히든 차원 (hidden_size): {config.vision_config.hidden_size}")
print(f"3. 최종 출력 차원 (projection_dim): {config.projection_dim}")
print(f"4. 입력 이미지 해상도: {config.vision_config.image_size}")