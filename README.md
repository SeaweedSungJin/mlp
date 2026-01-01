# SCOOP-RAG

# 실행
배치단위 X
```bash scripts/test_llava.sh```
배치 단위 동작 테스트 중
```bash scripts/test_rereanker_new.sh```

# 실험 설정
- 비전 인코더 선택: `config/retriever_config.yaml`에서 `type: eva_clip | clip_mlp`와 `projector_path` 변경
- 현재 파이프라인: 이미지 검색 → 섹션 분할 → Q-former rerank → top-1 섹션만 MLLM 컨텍스트로 사용
