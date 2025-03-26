from cluster_utils import main

if __name__ == "__main__":
    #def main(process_dfs = False, process_behaviors = False, data_dir = 'dataset/train/', valid_data_dir = 'dataset/valid/', zip_file = f"MINDlarge_train.zip", valid_zip_file = f"MINDlarge_dev.zip"):
    #main()
    main(False, False, 'dataset/small/train/', 'dataset/small/valid/', f"MINDsmall_train.zip", f"MINDsmall_dev.zip", 'small_user_category_profiles.pkl', 'small_user_cluster_df.pkl')

