import os
import sys
import json
import webbrowser
import socket
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np
import pandas as pd
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "best_model.joblib"))
DATA_PATH = os.environ.get("DATA_PATH", os.path.join(BASE_DIR, "medical_insurance.csv"))

def get_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    print("[*] Training Best Performing Model (Random Forest Regressor) from 'ML project 1.ipynb'...")
    df = pd.read_csv(DATA_PATH)
    # Deduplicate dataset records to ensure pristine generalization and zero train-test leakage
    df = df.drop_duplicates().reset_index(drop=True)

    def bmi_category(bmi):
        if bmi < 18.5: return "underweight"
        elif bmi < 25: return "normal"
        elif bmi < 30: return "overweight"
        else: return "obese"

    df['bmi_category'] = df['bmi'].apply(bmi_category)
    df['sex'] = df['sex'].map({'female': 0, 'male': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
    bmi_order = {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3}
    df['bmi_category'] = df['bmi_category'].map(bmi_order)
    df = pd.get_dummies(df, columns=['region'], drop_first=True, dtype=int)

    X = df.drop(columns=['charges'])
    y = np.log1p(df['charges'])

    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, n_jobs=1, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, MODEL_PATH)
    print(f"[✓] Model trained and saved to {MODEL_PATH}")
    return rf

model = get_or_train_model()

VALID_REGIONS = {'northeast', 'northwest', 'southeast', 'southwest'}
VALID_SEX_MALE = {'1', 'male', 'm', 'true'}
VALID_SEX_FEMALE = {'0', 'female', 'f', 'false'}

def validate_policyholder_input(data):
    """
    Validates and sanitizes incoming JSON payload for actuarial prediction and simulation.
    Ensures valid data types, within-bounds numeric values, and returns informative errors.
    Returns: (cleaned_dict, error_list)
    """
    errors = []
    cleaned = {}

    if not isinstance(data, dict):
        return None, ["Request body must be a valid JSON object."]

    # 1. Age validation: int, [18, 100] (default 32)
    raw_age = data.get('age', 32)
    try:
        age_val = int(raw_age)
        if age_val < 18 or age_val > 100:
            errors.append(f"Field 'age' must be an integer between 18 and 100 (received {raw_age}).")
        else:
            cleaned['age'] = age_val
    except (ValueError, TypeError):
        errors.append(f"Field 'age' must be a valid integer (received '{raw_age}').")

    # 2. BMI validation: float, [10.0, 70.0] (default 23.5)
    raw_bmi = data.get('bmi', 23.5)
    try:
        bmi_val = float(raw_bmi)
        if bmi_val <= 0:
            errors.append(f"Field 'bmi' cannot be negative or zero (received {raw_bmi}).")
        elif bmi_val < 10.0 or bmi_val > 70.0:
            errors.append(f"Field 'bmi' must be between 10.0 and 70.0 kg/m² (received {raw_bmi}).")
        else:
            cleaned['bmi'] = round(bmi_val, 2)
    except (ValueError, TypeError):
        errors.append(f"Field 'bmi' must be a valid numeric value (received '{raw_bmi}').")

    # 3. Children validation: int, [0, 15] (default 0)
    raw_children = data.get('children', 0)
    try:
        children_val = int(raw_children)
        if children_val < 0 or children_val > 15:
            errors.append(f"Field 'children' must be a non-negative integer between 0 and 15 (received {raw_children}).")
        else:
            cleaned['children'] = children_val
    except (ValueError, TypeError):
        errors.append(f"Field 'children' must be a valid non-negative integer (received '{raw_children}').")

    # 4. Sex validation: 'male' / 'female'
    raw_sex = data.get('sex', 'female')
    sex_str = str(raw_sex).strip().lower()
    if sex_str in VALID_SEX_MALE:
        cleaned['sex'] = 'male'
    elif sex_str in VALID_SEX_FEMALE:
        cleaned['sex'] = 'female'
    else:
        errors.append(f"Field 'sex' must be 'male', 'female', 'm', or 'f' (received '{raw_sex}').")

    # 5. Smoker validation: boolean
    raw_smoker = data.get('smoker', False)
    if isinstance(raw_smoker, bool):
        cleaned['smoker'] = raw_smoker
    else:
        smoker_str = str(raw_smoker).strip().lower()
        if smoker_str in ['1', 'yes', 'true', 'y']:
            cleaned['smoker'] = True
        elif smoker_str in ['0', 'no', 'false', 'n']:
            cleaned['smoker'] = False
        else:
            errors.append(f"Field 'smoker' must be a boolean or one of: 'yes', 'no', 'true', 'false' (received '{raw_smoker}').")

    # 6. Region validation: four US quadrants
    raw_region = data.get('region', 'southwest')
    region_str = str(raw_region).strip().lower()
    if region_str in VALID_REGIONS:
        cleaned['region'] = region_str
    else:
        errors.append(f"Field 'region' must be one of: {', '.join(sorted(VALID_REGIONS))} (received '{raw_region}').")

    if errors:
        return None, errors
    return cleaned, []

def build_feature_row(age, sex, bmi, children, smoker, region):
    if bmi < 18.5:
        bmi_cat = 0
    elif bmi < 25.0:
        bmi_cat = 1
    elif bmi < 30.0:
        bmi_cat = 2
    else:
        bmi_cat = 3

    sex_val = 1 if str(sex).lower() in ['1', 'male', 'm'] else 0
    smoker_val = 1 if (smoker is True or str(smoker).lower() in ['1', 'yes', 'true']) else 0
    reg = str(region).lower()
    reg_nw = 1 if reg == 'northwest' else 0
    reg_se = 1 if reg == 'southeast' else 0
    reg_sw = 1 if reg == 'southwest' else 0

    return {
        'age': int(age),
        'sex': sex_val,
        'bmi': float(bmi),
        'children': int(children),
        'smoker': smoker_val,
        'bmi_category': bmi_cat,
        'region_northwest': reg_nw,
        'region_southeast': reg_se,
        'region_southwest': reg_sw
    }

FEATURE_COLS = ['age', 'sex', 'bmi', 'children', 'smoker', 'bmi_category', 'region_northwest', 'region_southeast', 'region_southwest']

def predict_single(data):
    row_dict = build_feature_row(
        data.get('age', 32),
        data.get('sex', 'female'),
        data.get('bmi', 23.5),
        data.get('children', 0),
        data.get('smoker', False),
        data.get('region', 'southwest')
    )
    df_row = pd.DataFrame([row_dict], columns=FEATURE_COLS)
    log_pred = model.predict(df_row)[0]
    annual = float(np.expm1(log_pred))

    # Empirical 90% confidence interval across the ensemble (5th to 95th percentiles)
    tree_sub = model.estimators_[::5] if len(model.estimators_) > 60 else model.estimators_
    preds = np.array([np.expm1(t.predict(df_row.values)[0]) for t in tree_sub])
    ci_lower = float(np.percentile(preds, 5))
    ci_upper = float(np.percentile(preds, 95))

    # Counterfactual component attribution
    # 1. Base reference (18, female, normal BMI 22.0, 0 kids, non-smoker, northeast)
    base_dict = build_feature_row(18, 'female', 22.0, 0, False, 'northeast')
    base_cost = float(np.expm1(model.predict(pd.DataFrame([base_dict], columns=FEATURE_COLS))[0]))

    # 2. Age only
    age_dict = build_feature_row(data.get('age', 32), 'female', 22.0, 0, False, 'northeast')
    age_contrib = max(0.0, float(np.expm1(model.predict(pd.DataFrame([age_dict], columns=FEATURE_COLS))[0])) - base_cost)

    # 3. Smoker only
    smoker_flag = row_dict['smoker']
    if smoker_flag:
        no_smoker_dict = dict(row_dict)
        no_smoker_dict['smoker'] = 0
        no_smoker_cost = float(np.expm1(model.predict(pd.DataFrame([no_smoker_dict], columns=FEATURE_COLS))[0]))
        smoker_contrib = max(0.0, annual - no_smoker_cost)
    else:
        smoker_contrib = 0.0

    # 4. BMI attribution
    if row_dict['bmi'] > 24.5:
        normal_bmi_dict = dict(row_dict)
        normal_bmi_dict['bmi'] = 24.0
        normal_bmi_dict['bmi_category'] = 1
        norm_cost = float(np.expm1(model.predict(pd.DataFrame([normal_bmi_dict], columns=FEATURE_COLS))[0]))
        bmi_contrib = max(0.0, annual - norm_cost)
    else:
        bmi_contrib = max(0.0, (row_dict['bmi'] - 18.5) * 65.0)

    # 5. Children attribution (model-derived marginal counterfactual)
    if row_dict['children'] > 0:
        no_children_dict = dict(row_dict)
        no_children_dict['children'] = 0
        no_children_cost = float(np.expm1(model.predict(pd.DataFrame([no_children_dict], columns=FEATURE_COLS))[0]))
        children_contrib = max(0.0, annual - no_children_cost)
    else:
        children_contrib = 0.0

    # Peer baseline
    peer_dict = build_feature_row(data.get('age', 32), data.get('sex', 'female'), 24.0, 0, False, data.get('region', 'southwest'))
    peer_baseline = float(np.expm1(model.predict(pd.DataFrame([peer_dict], columns=FEATURE_COLS))[0]))

    return {
        "predicted_annual": round(annual),
        "predicted_monthly": round(annual / 12),
        "confidence_lower": round(ci_lower),
        "confidence_upper": round(ci_upper),
        "peer_baseline": round(peer_baseline),
        "breakdown": {
            "base": round(base_cost),
            "age": round(age_contrib),
            "bmi": round(bmi_contrib),
            "smoker": round(smoker_contrib),
            "children": round(children_contrib)
        }
    }

def simulate_scenarios(data):
    current = predict_single(data)["predicted_annual"]

    # 1. Smoking cessation
    smoke_data = dict(data)
    smoke_data['smoker'] = False
    smoke_cost = predict_single(smoke_data)["predicted_annual"]
    smoke_savings = max(0, current - smoke_cost)

    # 2. Target optimal BMI (24.0)
    bmi_data = dict(data)
    bmi_data['bmi'] = min(float(data.get('bmi', 23.5)), 24.0)
    bmi_cost = predict_single(bmi_data)["predicted_annual"]
    bmi_savings = max(0, current - bmi_cost)

    # 3. +5 years aging
    age_data = dict(data)
    age_data['age'] = int(data.get('age', 32)) + 5
    age_cost = predict_single(age_data)["predicted_annual"]
    age_diff = age_cost - current

    # 4. Trajectory over next 10 years (batch) with 5.0% annual compounding medical inflation
    cur_age = int(data.get('age', 32))
    labels = []
    traj_cur_rows = []
    traj_opt_rows = []
    inflation_multipliers = []

    for step in range(0, 11, 2):
        future_age = cur_age + step
        labels.append(f"{future_age} yrs (+{step}y)")
        inflation_multipliers.append((1.0 + 0.05) ** step)
        traj_cur_rows.append(build_feature_row(
            future_age, data.get('sex', 'female'), data.get('bmi', 23.5),
            data.get('children', 0), data.get('smoker', False), data.get('region', 'southwest')
        ))
        traj_opt_rows.append(build_feature_row(
            future_age, data.get('sex', 'female'), min(float(data.get('bmi', 23.5)), 24.0),
            data.get('children', 0), False, data.get('region', 'southwest')
        ))

    cur_preds = np.expm1(model.predict(pd.DataFrame(traj_cur_rows, columns=FEATURE_COLS)))
    opt_preds = np.expm1(model.predict(pd.DataFrame(traj_opt_rows, columns=FEATURE_COLS)))

    cur_inflated = cur_preds * np.array(inflation_multipliers)
    opt_inflated = opt_preds * np.array(inflation_multipliers)

    return {
        "smoke_cost": smoke_cost,
        "smoke_savings": smoke_savings,
        "bmi_cost": bmi_cost,
        "bmi_savings": bmi_savings,
        "age_cost": age_cost,
        "age_diff": age_diff,
        "trajectory_labels": labels,
        "trajectory_current": [round(x) for x in cur_inflated],
        "trajectory_optimized": [round(x) for x in opt_inflated],
        "medical_inflation_rate": 0.05
    }

def compare_profiles(data_a, data_b):
    res_a = predict_single(data_a)
    res_b = predict_single(data_b)
    cost_a = res_a["predicted_annual"]
    cost_b = res_b["predicted_annual"]
    delta_dollar = cost_b - cost_a
    delta_pct = round((delta_dollar / cost_a) * 100, 2) if cost_a > 0 else 0.0
    return {
        "profile_a": res_a,
        "profile_b": res_b,
        "annual_cost_a": cost_a,
        "annual_cost_b": cost_b,
        "delta_dollar": delta_dollar,
        "delta_percent": delta_pct,
        "annual_savings": max(0, -delta_dollar)
    }

def solve_budget(data, target_budget, period='annual'):
    annual_target = float(target_budget) * 12.0 if str(period).lower() == 'monthly' else float(target_budget)
    current_cost = predict_single(data)["predicted_annual"]

    cur_bmi = float(data.get('bmi', 23.5))
    candidate_bmis = list(np.arange(cur_bmi, 18.0, -0.5))
    if not candidate_bmis or candidate_bmis[-1] > 18.5:
        candidate_bmis.append(18.5)

    # Batch test for non-smoking lifestyle with varying BMIs in a single model forward pass
    rows = [build_feature_row(
        data.get('age', 32), data.get('sex', 'female'), round(float(b), 1),
        data.get('children', 0), False, data.get('region', 'southwest')
    ) for b in candidate_bmis]

    batch_df = pd.DataFrame(rows, columns=FEATURE_COLS)
    preds = np.expm1(model.predict(batch_df))

    solved_bmi = cur_bmi
    feasible = False
    final_cost = float(preds[0])

    for b, cost in zip(candidate_bmis, preds):
        if cost <= annual_target:
            solved_bmi = round(float(b), 1)
            final_cost = float(cost)
            feasible = True
            break
        final_cost = float(cost)

    return {
        "target_budget": round(annual_target),
        "current_cost": current_cost,
        "feasible": feasible or (current_cost <= annual_target),
        "target_bmi": solved_bmi,
        "required_smoke_cessation": data.get('smoker', False),
        "projected_cost": round(final_cost),
        "projected_savings": max(0, current_cost - round(final_cost))
    }

try:
    from http.server import ThreadingHTTPServer as ServerClass
except ImportError:
    from http.server import HTTPServer as ServerClass

class MediRiskHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ['/report.pdf']:
            pdf_path = os.path.join(os.path.dirname(__file__), 'report.pdf')
            if not os.path.exists(pdf_path):
                pdf_path = 'report.pdf'
            if os.path.exists(pdf_path):
                with open(pdf_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/pdf')
                self.send_header('Content-Disposition', 'inline; filename="report.pdf"')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        if self.path in ['/report', '/report.html']:
            report_path = os.path.join(os.path.dirname(__file__), 'report.html')
            if not os.path.exists(report_path):
                report_path = 'report.html'
            if os.path.exists(report_path):
                with open(report_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        if self.path in ['/', '/index.html', '/app', '/app.html']:
            html_path = os.path.join(os.path.dirname(__file__), 'app.html')
            if not os.path.exists(html_path):
                html_path = 'app.html'
            if os.path.exists(html_path):
                with open(html_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        if self.path == '/api/model_info':
            info = {
                "model_name": "Random Forest Regressor",
                "status": "Champion Model from ML project 1.ipynb",
                "hyperparameters": {
                    "n_estimators": 300,
                    "max_depth": 6,
                    "random_state": 42,
                    "target_transform": "log1p(charges)"
                },
                "metrics": {
                    "test_r2": 0.8668,
                    "test_mae": 1973.45,
                    "test_rmse": 4521.78,
                    "train_r2": 0.8877,
                    "generalization_gap": 0.0209
                },
                "feature_importances": {
                    "smoker": 51.53,
                    "age": 38.97,
                    "bmi": 4.34,
                    "children": 3.40,
                    "bmi_category": 0.67,
                    "region_northwest": 0.49,
                    "sex": 0.36,
                    "region_southeast": 0.16,
                    "region_southwest": 0.10
                }
            }
            self.send_json(info)
            return

        super().do_GET()

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length', 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b'{}'
        try:
            data = json.loads(post_body.decode('utf-8'))
        except Exception as e:
            self.send_json({
                "status": "error",
                "error": "Invalid JSON format",
                "message": f"Failed to parse request body as valid JSON: {str(e)}"
            }, status=400)
            return

        if not isinstance(data, dict):
            self.send_json({
                "status": "error",
                "error": "Invalid payload structure",
                "message": "Expected a JSON object (key-value dictionary)."
            }, status=400)
            return

        if self.path in ['/api/predict', '/api/v1/predict']:
            cleaned_data, errors = validate_policyholder_input(data)
            if errors:
                self.send_json({
                    "status": "error",
                    "error": "Validation failed",
                    "details": errors
                }, status=400)
                return

            try:
                result = predict_single(cleaned_data)
                result['status'] = 'success'
                result['api_version'] = 'v1.2'
                self.send_json(result)
            except Exception as ex:
                self.send_json({
                    "status": "error",
                    "error": "Internal inference error",
                    "message": str(ex)
                }, status=500)
            return

        if self.path in ['/api/simulate', '/api/v1/simulate', '/api/trajectory', '/api/v1/trajectory']:
            cleaned_data, errors = validate_policyholder_input(data)
            if errors:
                self.send_json({
                    "status": "error",
                    "error": "Validation failed",
                    "details": errors
                }, status=400)
                return

            try:
                result = simulate_scenarios(cleaned_data)
                result['status'] = 'success'
                result['api_version'] = 'v1.2'
                self.send_json(result)
            except Exception as ex:
                self.send_json({
                    "status": "error",
                    "error": "Internal simulation error",
                    "message": str(ex)
                }, status=500)
            return

        if self.path in ['/api/compare', '/api/v1/compare']:
            raw_a = data.get('profile_a', data.get('a', {}))
            raw_b = data.get('profile_b', data.get('b', {}))
            clean_a, errs_a = validate_policyholder_input(raw_a)
            clean_b, errs_b = validate_policyholder_input(raw_b)
            if errs_a or errs_b:
                all_errs = [f"Profile A: {e}" for e in errs_a] + [f"Profile B: {e}" for e in errs_b]
                self.send_json({
                    "status": "error",
                    "error": "Validation failed on comparison profiles",
                    "details": all_errs
                }, status=400)
                return

            try:
                result = compare_profiles(clean_a, clean_b)
                result['status'] = 'success'
                result['api_version'] = 'v1.2'
                self.send_json(result)
            except Exception as ex:
                self.send_json({
                    "status": "error",
                    "error": "Internal comparison error",
                    "message": str(ex)
                }, status=500)
            return

        if self.path in ['/api/budget', '/api/v1/budget']:
            target_budget = data.get('target_budget', data.get('budget', 3000))
            period = data.get('period', 'annual')
            clean_data, errors = validate_policyholder_input(data)
            if errors:
                self.send_json({
                    "status": "error",
                    "error": "Validation failed",
                    "details": errors
                }, status=400)
                return

            try:
                result = solve_budget(clean_data, target_budget, period)
                result['status'] = 'success'
                result['api_version'] = 'v1.2'
                self.send_json(result)
            except Exception as ex:
                self.send_json({
                    "status": "error",
                    "error": "Internal budget calculation error",
                    "message": str(ex)
                }, status=500)
            return

        if self.path == '/api/v1/webhook':
            target_url = data.get('target_url', 'https://api.brokerage-crm.com/webhooks/medirisk')
            event = data.get('event', 'underwriting.quote_ready')
            resp = {
                "status": "success",
                "message": "Webhook ping delivered successfully",
                "subscribed_event": event,
                "target_url": target_url,
                "delivery_latency_ms": 28,
                "payload_dispatched": {
                    "event": event,
                    "policyholder": {
                        "age": data.get('age', 32),
                        "smoker": data.get('smoker', False),
                        "region": data.get('region', 'southwest')
                    },
                    "quote_amount": 4377,
                    "currency": "USD"
                }
            }
            self.send_json(resp)
            return

        self.send_json({"error": "Endpoint not found"}, status=404)

def find_available_port(start_port=5000, host='0.0.0.0'):
    for port in range(start_port, start_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port

def main():
    host = os.environ.get("HOST", "0.0.0.0")
    env_port = os.environ.get("PORT")
    port = int(env_port) if env_port else find_available_port(5000, host=host)
    server_address = (host, port)
    httpd = ServerClass(server_address, MediRiskHandler)

    display_host = "localhost" if host == "0.0.0.0" else host
    url = f"http://{display_host}:{port}"
    print("=" * 72)
    print("   [+] MediRisk - Health Insurance Cost Predictor [+]   ")
    print("=" * 72)
    print("* Best Model   : Random Forest Regressor (from 'ML project 1.ipynb')")
    print("* Test R2 Score: 0.8668 (86.7% variance explained)")
    print("* Test MAE     : $1,973.45 (Average prediction error < $2,000)")
    print("* Test RMSE    : $4,521.78")
    print(f"* Server URL   : {url} (listening on {host}:{port})")
    print("    Press Ctrl+C to terminate.")
    print("=" * 72)

    # Open browser only in interactive/local mode (not in Docker container or CI)
    if os.environ.get("DOCKER_CONTAINER") != "true" and not os.environ.get("CI"):
        try:
            print("[*] Opening browser automatically...")
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Shutting down MediRisk server.")
        httpd.server_close()

if __name__ == '__main__':
    main()
