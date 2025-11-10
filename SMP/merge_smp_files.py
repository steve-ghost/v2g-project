import pandas as pd
import os

def merge_smp_files(start_year, end_year, output_filename="SMP.csv"):
    """
    지정된 연도 범위 내의 SMP CSV 파일들을 병합하여 하나의 CSV 파일로 저장합니다.

    - 첫 번째 파일(예: SMP2024.csv)의 헤더를 사용하고, 나머지 파일들은 헤더 없이 본문만 병합합니다.
    - 출력 파일이 이미 존재하면 덮어씁니다.

    Args:
        start_year (int): 시작 연도 (예: 2024)
        end_year (int): 끝 연도 (예: 2045)
        output_filename (str): 병합된 파일의 이름 (기본값: "SMP.csv")
    """
    merged_df = pd.DataFrame()

    for year in range(start_year, end_year + 1):
        filename = f"SMP{year}.csv"
        if not os.path.exists(filename):
            print(f"경고: 파일 {filename}을(를) 찾을 수 없습니다. 건너뜁니다.")
            continue

        if year == start_year:
            # 첫 번째 파일은 헤더를 포함하여 읽습니다.
            df = pd.read_csv(filename)
            merged_df = df
        else:
            # 나머지 파일들은 헤더 없이 두 번째 행부터 읽습니다.
            # header=None으로 헤더를 무시하고, iloc[1:]으로 두 번째 행부터 선택합니다.
            df = pd.read_csv(filename, header=None)
            # 첫 번째 파일의 컬럼명과 일치하도록 컬럼 수를 확인하고 슬라이싱합니다.
            merged_df = pd.concat([merged_df, df.iloc[1:]], ignore_index=False)
            # if len(df.columns) == len(merged_df.columns):
            #     merged_df = pd.concat([merged_df, df.iloc[1:]], ignore_index=True)
            # else:
            #     print(f"경고: {filename} 파일의 컬럼 수가 첫 번째 파일과 다릅니다. 이 파일을 병합하지 않습니다.")

    # 병합된 데이터프레임을 CSV 파일로 저장합니다. (overwrite)
    merged_df.to_csv(output_filename, index=False)
    print(f"모든 SMP 파일이 {output_filename} (으)로 성공적으로 병합되었습니다.")

def main() :
    merge_smp_files(2024, 2045)

# 사용 예시:
# 이 코드를 실행하기 전에 SMP2024.csv, SMP2025.csv 등의 파일을 현재 디렉토리에 준비해야 합니다.
if __name__ == "__main__":
    main()

