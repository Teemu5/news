from cluster_utils import main, run_meta_training
import argparse

if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Run NRS evaluation for specified cluster(s).")
     parser.add_argument("--dataset", type=str, default="train", help="Dataset type: train or valid")
     parser.add_argument("--cluster_id", type=str, default=None,
                         help="Cluster ID (or comma-separated list) to evaluate (if not provided, all clusters will be processed)")
     parser.add_argument("--dataset_size", type=str, default="large", help="Dataset size: large or small")
     parser.add_argument("--process_dfs", action='store_true', help="Process dataframes if needed")
     parser.add_argument("--process_behaviors", action='store_true', help="Process behaviors if needed")
     args = parser.parse_args()
     if args.dataset_size == "small":
         data_dir_train="dataset/small/train/"
         data_dir_valid="dataset/small/valid/"
         zip_file_train="MINDsmall_train.zip"
         zip_file_valid="MINDsmall_dev.zip"
         user_category_profiles_path="small_user_category_profiles.pkl"
         user_cluster_df_path="small_user_cluster_df.pkl"
         cluster_id=args.cluster_id
     else:
         data_dir_train="dataset/train/"
         data_dir_valid="dataset/valid/"
         zip_file_train="MINDlarge_train.zip"
         zip_file_valid="MINDlarge_dev.zip"
         user_category_profiles_path="user_category_profiles.pkl"
         user_cluster_df_path="user_cluster_df.pkl"
         cluster_id=args.cluster_id

     run_meta_training(dataset=args.dataset, dataset_size=args.dataset_size,
         process_dfs=args.process_dfs, process_behaviors=args.process_behaviors,
         data_dir_train=data_dir_train, data_dir_valid=data_dir_valid,
         zip_file_train=zip_file_train, zip_file_valid=zip_file_valid,
         user_category_profiles_path=user_category_profiles_path,
         user_cluster_df_path=user_cluster_df_path, cluster_id=args.cluster_id)
     """main(dataset=args.dataset,
          process_dfs=args.process_dfs, process_behaviors=args.process_behaviors,
          data_dir_train="dataset/train/", data_dir_valid="dataset/valid/",
          zip_file_train="MINDlarge_train.zip", zip_file_valid="MINDlarge_dev.zip",
          user_category_profiles_path="user_category_profiles.pkl",
          user_cluster_df_path="user_cluster_df.pkl", cluster_id=args.cluster_id)
     """
     #def main(process_dfs = False, process_behaviors = False, data_dir = 'dataset/train/', valid_data_dir = 'dataset/valid/', zip_file = f"MINDlarge_train.zip", valid_zip_file = f"MINDlarge_dev.zip"):
     #main()
     #main(False, False, 'dataset/small/train/', 'dataset/small/valid/', f"MINDsmall_train.zip", f"MINDsmall_dev.zip", 'small_user_category_profiles.pkl', 'small_user_cluster_df.pkl')
     #main('valid', False, False, 'dataset/train/', 'dataset/valid/', f"MINDlarge_train.zip", f"MINDlarge_dev.zip", 'user_category_profiles.pkl', 'user_cluster_df.pkl')

