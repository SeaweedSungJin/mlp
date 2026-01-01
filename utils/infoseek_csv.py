# import pandas as pd

# def sample_csv_by_image_id(input_csv, output_csv, n=3, seed=42):
#     df = pd.read_csv(input_csv)
#     original_count = len(df)

#     def safe_sample(group):
#         k = min(len(group), n)  # 그룹 크기보다 많지 않게 설정
#         return group.sample(k, random_state=seed)

#     sampled = df.groupby("wikipedia_title", group_keys=False).apply(safe_sample)
#     sampled_count = len(sampled)

#     sampled.to_csv(output_csv, index=False)

#     print("=== Sampling Summary ===")
#     print(f"Original CSV: {original_count} rows")
#     print(f"Sampled CSV:  {sampled_count} rows")
#     print(f"Reduced by:    {original_count - sampled_count} rows")
#     print(f"Saved sampled CSV to {output_csv}")

# # 사용 예시
# sample_csv_by_image_id(
#     "/data/dataset/infoseek/infoseek_test_filtered.csv",
#     "/data/dataset/infoseek/infoseek_sampled_wikititle.csv",
#     n=3
# )

import os
import csv
import pandas as pd

def get_image(image_id, dataset_name, iNat_id2name=None):

    # if dataset_name == "inaturalist":
    #     file_name = iNat_id2name.get(image_id)
    #     if file_name:
    #         return os.path.join(iNat_image_path, file_name)
    #     return None

    # elif dataset_name == "landmarks":
    #     try:
    #         return _resolve_landmark_image(image_id)
    #     except:
    #         return None

    # elif dataset_name == "infoseek":
    category = image_id.split("_")[0]
    base = "/dataset/infoseek"

    candidates = [
        os.path.join(base, category, "images", image_id + ".jpg"),
        os.path.join(base, category, "images", image_id + ".jpeg"),
        os.path.join(base, category, "images", image_id + ".JPEG"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None

# ===================================================================
# ✔ test.csv에서 이미지 없는 데이터 개수 세기 + 필터링 CSV 생성
# ===================================================================

def filter_missing_images(
    csv_path: str,
    output_csv_path: str,
    iNat_id2name: dict = None
):
    df = pd.read_csv(csv_path)

    missing_rows = []
    valid_rows = []

    for idx, row in df.iterrows():
        image_id = str(row["dataset_image_ids"])
        dataset_name = "infoseek"

        img_path = get_image(image_id, dataset_name, iNat_id2name)

        if img_path is None or not os.path.exists(img_path):
            missing_rows.append(row)
        else:
            valid_rows.append(row)

    # 결과 출력
    print(f"전체 샘플 수       : {len(df)}")
    print(f"이미지 없는 샘플 수 : {len(missing_rows)}")
    print(f"유효한 샘플 수     : {len(valid_rows)}")

    # 유효한 데이터만 저장
    valid_df = pd.DataFrame(valid_rows)
    valid_df.to_csv(output_csv_path, index=False)
    print(f"이미지 존재 샘플만 {output_csv_path} 에 저장 완료.")

    return missing_rows, valid_rows


# -----------------------
# 사용 예시
# -----------------------
if __name__ == "__main__":
    # 만약 inaturalist 매핑 JSON이 있다면 미리 로드
    # iNat_id2name = json.load(open("train_id2name.json"))
    iNat_id2name = None

    filter_missing_images(
        csv_path="/dataset/infoseek/infoseek_test_filtered.csv",
        output_csv_path="/dataset/infoseek/infoseek_test_new_filtered.csv",
        iNat_id2name=iNat_id2name
    )
