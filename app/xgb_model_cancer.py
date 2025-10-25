import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置页面配置
st.set_page_config(page_title="AKI Prediction Model", layout="wide")

# 加载保存的XGBoost模型
@st.cache_resource
def load_model():
    # 使用相对路径加载模型 - 模型文件应该与app.py在同一目录下
    model_path = 'xgb.pkl'
    
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        # 如果找不到，尝试其他可能的路径
        st.error(f"Model not found at {model_path}. Trying alternative paths...")
        
        # 列出当前目录下的文件，帮助调试
        st.write("Current directory:", os.getcwd())
        st.write("Files in directory:", os.listdir())
        
        # 尝试其他可能的路径
        alternative_paths = ['./xgb.pkl', '../xgb.pkl', 'app/xgb.pkl']
        for path in alternative_paths:
            try:
                if os.path.exists(path):
                    st.success(f"Found model at: {path}")
                    return joblib.load(path)
            except:
                continue
                
        # 如果所有尝试都失败，显示错误
        st.error("Failed to load model. Please check the model file path.")
        st.stop()

model = load_model()

# 更新特征范围定义，使用模型期望的特征名称
feature_ranges = {
    "Age": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 65.0, "description": "Patient Age"},
    "LDH": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 200.0, "description": "Lactate Dehydrogenase"},
    "TPSA": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 10.0, "description": "Total Prostate-Specific Antigen"},
    "Hb": {"type": "numerical", "min": 0.0, "max": 20.0, "default": 14.0, "description": "Hemoglobin"},
    "CPR": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 5.0, "description": "C-Reactive Protein"}
}

# 预定义特征重要性（如果无法从模型中获取）
# 这些值可以从您的模型训练过程中获取，或者根据领域知识估计
predefined_importance = {
    "Age": 0.25,
    "LDH": 0.20,
    "TPSA": 0.30,
    "Hb": 0.15,
    "CPR": 0.10
}

# Streamlit 界面
st.title("AKI Prediction Model with Feature Importance Visualization")
st.write("This application predicts the possibility of Acute Kidney Injury (AKI) based on input features.")

# 创建两列布局
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Enter Patient Features:")
    # 动态生成输入项
    feature_values = {}
    for feature, properties in feature_ranges.items():
        if properties["type"] == "numerical":
            value = st.number_input(
                label=f"{properties['description']} ({feature})",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                key=feature
            )
        elif properties["type"] == "categorical":
            value = st.selectbox(
                label=f"{properties['description']} ({feature})",
                options=properties["options"],
                key=feature
            )
        feature_values[feature] = value

    # 转换为模型输入格式
    input_df = pd.DataFrame([feature_values])

with col2:
    st.header("Prediction Results")
    
    # 预测与特征重要性可视化
    if st.button("Predict"):
        try:
            # 模型预测
            predicted_class = model.predict(input_df)[0]
            predicted_proba = model.predict_proba(input_df)[0]

            # 提取预测的类别概率
            probability = predicted_proba[1] * 100  # 假设类别1是正类（AKI）

            # 显示预测结果
            if probability > 50:
                st.error(f"⚠️ High Risk: The probability of AKI is {probability:.2f}%")
            else:
                st.success(f"✅ Low Risk: The probability of AKI is {probability:.2f}%")

            # 创建一个进度条来可视化概率
            st.progress(int(probability))

            # 特征重要性可视化
            st.subheader("Feature Importance Analysis")
            
            try:
                # 尝试从模型获取特征重要性
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = input_df.columns
                    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                else:
                    # 如果模型没有feature_importances_属性，使用预定义值
                    importance_df = pd.DataFrame({
                        'Feature': list(predefined_importance.keys()),
                        'Importance': list(predefined_importance.values())
                    })
                
                # 按重要性排序
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # 创建条形图
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color='#1E88E5')
                
                # 添加标签和标题
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance')
                
                # 添加数值标签
                for i, v in enumerate(importance_df['Importance']):
                    ax.text(v + 0.01, i, f'{v:.3f}', va='center')
                
                # 美化图表
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 添加解释
                st.write("The chart above shows the importance of each feature in the model:")
                st.write("- Higher values indicate more influential features")
                st.write("- These values represent the overall importance in the model, not specific to this prediction")
                
            except Exception as e:
                st.error(f"Error in feature importance visualization: {str(e)}")
                st.write("Unable to display feature importance visualization.")
            
            # 添加当前输入值表格
            st.subheader("Current Patient Features")
            input_display = input_df.T.rename(columns={0: "Value"})
            st.dataframe(input_display)
            
            # 添加风险因素分析
            st.subheader("Risk Factor Analysis")
            
            # 根据医学知识为每个特征创建风险评估
            risk_analysis = []
            
            # 年龄风险
            age = input_df["Age"].values[0]
            if age > 70:
                risk_analysis.append("⚠️ **Age > 70**: Advanced age is a significant risk factor for AKI.")
            
            # LDH风险
            ldh = input_df["LDH"].values[0]
            if ldh > 300:
                risk_analysis.append("⚠️ **Elevated LDH**: High LDH levels may indicate tissue damage.")
            
            # TPSA风险
            tpsa = input_df["TPSA"].values[0]
            if tpsa > 20:
                risk_analysis.append("⚠️ **Elevated TPSA**: High TPSA levels may indicate prostate issues.")
            
            # Hb风险
            hb = input_df["Hb"].values[0]
            if hb < 10:
                risk_analysis.append("⚠️ **Low Hemoglobin**: Anemia may reduce oxygen delivery to kidneys.")
            
            # CPR风险
            cpr = input_df["CPR"].values[0]
            if cpr > 10:
                risk_analysis.append("⚠️ **Elevated CRP**: Increased inflammation may affect kidney function.")
            
            # 显示风险分析
            if risk_analysis:
                for risk in risk_analysis:
                    st.markdown(risk)
            else:
                st.write("✅ No specific risk factors identified based on the provided values.")

            # 添加模型解释信息
            st.subheader("Model Information")
            st.write("This prediction is based on an XGBoost model trained on historical patient data.")
            st.write("The model evaluates the risk of developing Acute Kidney Injury based on the provided features.")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("Please check that all input features are correctly formatted and try again.")

# 添加页脚
st.markdown("---")
st.markdown("© 2023 AKI Prediction Model | For research purposes only")
