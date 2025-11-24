from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 🔹 1️⃣ 모델 불러오기
mlp_cl = joblib.load("mlp_cl.joblib")        # Cl 예측 모델
xgb_cd = joblib.load("xgb_cd.joblib")        # Cd 예측 모델
x_scaler = joblib.load("x_scaler.joblib")    # 입력 스케일러
y_scaler = joblib.load("y_cl_scaler.joblib") # Cl 출력 스케일러

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        aoa = float(request.form['aoa'])  # AOA 입력 받기
        X_input = np.array([[aoa]])

        # 입력 스케일링
        X_scaled = x_scaler.transform(X_input)

        # 예측
        cl_pred = mlp_cl.predict(X_scaled)
        cd_pred = xgb_cd.predict(X_scaled)

        # 스케일링 복원 (Cl만)
        cl_pred_rescaled = y_scaler.inverse_transform(cl_pred.reshape(-1, 1))
        cl_final = float(cl_pred_rescaled[0])
        cd_final = float(cd_pred[0])

        # ================================
        #    🔥 Downforce & DragForce 계산
        # ================================
        rho = 1.225      # 공기 밀도 (kg/m³)
        V   = 24.17      # 속도 고정 (m/s)
        S   = 1.0        # 기준 면적 (m²)

        q = 0.5 * rho * (V ** 2)   # 동압

        downforce = q * S * cl_final
        dragforce = q * S * cd_final
        # ================================

        result = {
            "AOA": aoa,
            "Cl": round(cl_final, 4),
            "Cd": round(cd_final, 4),
            "Downforce": round(downforce, 2),
            "Dragforce": round(dragforce, 2)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
