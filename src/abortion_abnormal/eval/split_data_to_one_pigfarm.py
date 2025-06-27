import pandas as pd
import os

class split_data_to_one_pigfarm:
     def __init__(self, pigfarm_dks, version, predict_data_path):
          self.pigfarm_dks = pigfarm_dks
          self.version = version
          self.predict_data = pd.read_csv(predict_data_path)

     def split_data(self):
          # Create output directory if it doesn't exist
          for pigfarm_dk in self.pigfarm_dks:
               output_dir = os.path.join('data', 'predict', 'abort_abnormal', f'{self.version} {pigfarm_dk}', f'{self.version} {pigfarm_dk} 12')
               os.makedirs(output_dir, exist_ok=True)

               # Filter data for current pigfarm_dk
               pdk = pigfarm_dk.replace('@', '/') # 替换回原来的斜杠格式
               filtered_data = self.predict_data[self.predict_data['pigfarm_dk'] == pdk]

               # Save filtered data to CSV
               output_path = os.path.join(output_dir, 'abort_abnormal.csv')
               filtered_data.to_csv(output_path, index=False)