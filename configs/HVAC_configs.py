from configs.common_configs import args

args.model_comment = 'HVAC'  # 能源供应
args.features = 'M'

'''
'Cumulative_Chiller_Energy_Consumption','Cumulative_Primary_Chilled_Water_Pump_Energy', 'Cumulative_Secondary_Chilled_Water_Pump_Energy', 'Cumulative_Cooling_Water_Pump_Energy ','Cumulative_Cooling_Tower_Energy', 
'Process_Chilled_Water_Pressure_Difference', 'Process_Chilled_Water_Supply_Pressure', 'Process_Chilled_Water_Supply_Temperature','Chilled_Water_Bypass_Temperature',
'AC_Chilled_Water_Supply_Temperature',  'AC_Chilled_Water_Bypass_Valve_Opening', 'Cooling_Water_Bypass_Valve_Opening', 'AC_Chilled_Water_Supply_Pressure', 'AC_Chilled_Water_Pressure_Difference',
 'Chilled_Water_Return_Pressure',  'Chilled_Water_Temperature_Difference', 'Chilled_Water_Distribution_Coefficient', 'Humidity','Process_Chilled_Water_Bypass_Valve_Opening',
'''
args.data = 'hvac'
args.root_path = './dataset/HVAC'
args.data_path = 'summary_2.csv'
args.feature_cols = ['Temperature', 'Dewpoint','Dry_Bulb_Temperature', 'Wet_Bulb_Temperature', 'Total_Chiller_Power', 'Cooling_Tower_Total_Power', 'Cooling_Water_Pump_Total_Power',
                     'Total_Chilled_Water_Flowrate', 'Process_Chilled_Water_Flowrate',  'AC_Chilled_Water_Flowrate', 'Cooling_Water_Return_Temperature', 'Chiller_COP',
                     'Chilled_Water_Supply_Pressure', 'Chilled_Water_Pressure_Difference', 'Cooling_Water_Supply_Temperature',  'Secondary_Chilled_Water_Pump_Total_Power_Process',
                     'Primary_Chilled_Water_Pump_Total_Power', 'Total_Cooling_Capacity',
                     'Chilled_Water_Pump_Efficiency',   'Chilled_Water_Bypass_Temperature', 'Chilled_Water_Return_Temperature', 'Chilled_Water_Supply_Temperature',
                     'Total_Power', 'System_COP', 'System_Energy_Efficiency']

#

args.target = ['Total_Power','Total_Chiller_Power','System_Energy_Efficiency','Total_Cooling_Capacity']
args.num_train = 35136  # 2023/03/08 - 2024/03/07
args.num_test = 15840  # # 2024/03/08
args.seq_len = 48
args.pred_len = 48
args.label_len = args.seq_len


args.c_out = len(args.target)
args.forecast_dim = 1
args.source_data_path = args.data_path
args.enc_in = len(args.feature_cols)
args.dec_in = len(args.feature_cols)

args.scale = True
args.val = False

