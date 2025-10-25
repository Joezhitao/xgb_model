import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 设置页面配置
st.set_page_config(page_title="AKI Prediction Model", layout="wide")

# 加载保存的XGBoost模型
@st.cache_resource
def load_model():
    return joblib.load('xgb.pkl')  # 使用相对路径

model = load_model()

# 更新特征范围定义，使用模型期望的特征名称
feature_ranges = {
    "Age": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 65.0, "description": "Patient Age"},
    "LDH": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 200.0, "description": "Lactate Dehydrogenase"},
    "TPSA": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 10.0, "description": "Total Prostate-Specific Antigen"},
    "Hb": {"type": "numerical", "min": 0.0, "max": 20.0, "default": 14.0, "description": "Hemoglobin"},
    "CPR": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 5.0, "description": "C-Reactive Protein"}
}

# Streamlit 界面
st.title("AKI Prediction Model with SHAP Visualization")
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
    
    # 预测与 SHAP 可视化
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

            # 计算 SHAP 值
            st.subheader("Feature Contribution Analysis")
            with st.spinner("Calculating SHAP values..."):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                
                # 检查shap_values的结构
                if isinstance(shap_values, list):
                    # 对于多类别问题，选择类别1（正类）的SHAP值
                    shap_values_to_plot = shap_values[1][0]
                    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                else:
                    # 对于二分类问题，直接使用返回的SHAP值
                    shap_values_to_plot = shap_values[0]
                    expected_value = explainer.expected_value

                # 创建条形图
                fig, ax = plt.subplots(figsize=(10, 6))
                features = input_df.columns
                y_pos = np.arange(len(features))
                
                # 根据SHAP值的正负设置颜色
                colors = ['#FF4B4B' if x > 0 else '#1E88E5' for x in shap_values_to_plot]
                
                # 绘制水平条形图
                bars = ax.barh(y_pos, shap_values_to_plot, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_xlabel('SHAP Value (Impact on Prediction)')
                ax.set_title('Feature Contributions to AKI Risk')
                
                # 添加基准线
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # 添加数值标签
                for i, v in enumerate(shap_values_to_plot):
                    if v >= 0:
                        ax.text(v + 0.001, i, f'+{v:.3f}', va='center', fontweight='bold')
                    else:
                        ax.text(v - 0.025, i, f'{v:.3f}', va='center', fontweight='bold')
                
                # 添加图例
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#FF4B4B', label='Increases Risk'),
                    Patch(facecolor='#1E88E5', label='Decreases Risk')
                ]
                ax.legend(handles=legend_elements, loc='lower right')
                
                # 美化图表
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 添加SHAP值解释
                st.write("The chart above shows how each feature contributes to the prediction:")
                st.write("- Red bars indicate features that increase the risk of AKI")
                st.write("- Blue bars indicate features that decrease the risk of AKI")
                st.write(f"- The base value (expected value) is: {expected_value:.4f}")
                
                # 添加特征值表格
                st.subheader("Current Patient Features")
                st.dataframe(input_df.T.rename(columns={0: "Value"}))

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
