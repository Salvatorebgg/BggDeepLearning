# -*- coding: utf-8 -*-

"""
BggDeepLearning API 测试客户端脚本

File:
scripts/python/test_api_client.py

Purpose:
- 自动测试 FastAPI 服务的所有端点
- 验证健康检查、单预测、批量预测、报告导出

Usage:
# 先启动 API 服务（另一个终端）
python apps/api/main.py

# 然后运行测试
python scripts/python/test_api_client.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test BggDeepLearning FastAPI endpoints."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="API base URL. Default: http://localhost:8000",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default="random_forest",
        help="Model key to test with. Default: random_forest",
    )
    return parser.parse_args()


def api_request(
    base_url: str,
    endpoint: str,
    method: str = "GET",
    data: dict | None = None,
) -> tuple[int, dict | bytes]:
    """
    Make an API request and return (status_code, response_data).
    """
    url = f"{base_url}{endpoint}"
    headers = {"Content-Type": "application/json"}

    body = None
    if data is not None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")

    req = Request(url, data=body, headers=headers, method=method)

    try:
        with urlopen(req, timeout=30) as response:
            content_type = response.headers.get("Content-Type", "")
            raw = response.read()

            if "application/json" in content_type:
                return response.status, json.loads(raw.decode("utf-8"))
            else:
                return response.status, raw

    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8", errors="ignore"))
    except URLError as exc:
        print(f"  连接失败: {exc.reason}")
        print(f"  请确认 API 服务已启动: python apps/api/main.py")
        sys.exit(1)


def test_health(base_url: str) -> bool:
    """Test health check endpoint."""
    print("1. 健康检查 GET /api/v1/health ... ", end="")
    status, data = api_request(base_url, "/api/v1/health")
    if status == 200 and isinstance(data, dict) and data.get("status") == "ok":
        print(f"PASS (models: {data.get('available_models', [])})")
        return True
    print(f"FAIL status={status}")
    return False


def test_list_models(base_url: str) -> bool:
    """Test models listing endpoint."""
    print("2. 模型列表 GET /api/v1/models ... ", end="")
    status, data = api_request(base_url, "/api/v1/models")
    if status == 200 and isinstance(data, dict) and "models" in data:
        model_keys = [m["model_key"] for m in data["models"]]
        print(f"PASS ({len(model_keys)} models: {model_keys})")
        return True
    print(f"FAIL status={status}")
    return False


def test_single_prediction(base_url: str, model_key: str) -> bool:
    """Test single patient prediction."""
    print(f"3. 单个预测 POST /api/v1/predict/single?model_key={model_key} ... ", end="")

    payload = {
        "age": 65,
        "sex": "Male",
        "bmi": 24.0,
        "admission_type": "Emergency",
        "hypertension": 1,
        "diabetes": 0,
        "coronary_disease": 0,
        "chronic_kidney_disease": 0,
        "infection_suspected": 1,
        "trauma_suspected": 0,
        "heart_rate": 105,
        "systolic_bp": 105,
        "diastolic_bp": 65,
        "respiratory_rate": 24,
        "oxygen_saturation": 93,
        "temperature_c": 38.0,
        "hemoglobin": 11.0,
        "white_blood_cell": 13.0,
        "platelet": 180.0,
        "creatinine": 120.0,
        "lactate": 3.0,
        "c_reactive_protein": 80.0,
    }

    status, data = api_request(
        base_url,
        f"/api/v1/predict/single?model_key={model_key}",
        method="POST",
        data=payload,
    )

    if status == 200 and isinstance(data, dict):
        prob = data.get("probability", "N/A")
        risk = data.get("risk_group", "N/A")
        expl = len(data.get("explanation", []))
        print(f"PASS prob={prob:.4f}, risk={risk}, expl={expl} items")
        return True

    print(f"FAIL status={status}, detail={data.get('detail', data)}")
    return False


def test_batch_prediction(base_url: str, model_key: str) -> bool:
    """Test batch prediction."""
    print(f"4. 批量预测 POST /api/v1/predict/batch?model_key={model_key} ... ", end="")

    patients = [
        {
            "age": 65, "sex": "Male", "bmi": 24.0, "admission_type": "Emergency",
            "hypertension": 1, "diabetes": 0, "coronary_disease": 0,
            "chronic_kidney_disease": 0, "infection_suspected": 1, "trauma_suspected": 0,
            "heart_rate": 105, "systolic_bp": 105, "diastolic_bp": 65,
            "respiratory_rate": 24, "oxygen_saturation": 93, "temperature_c": 38.0,
            "hemoglobin": 11.0, "white_blood_cell": 13.0, "platelet": 180.0,
            "creatinine": 120.0, "lactate": 3.0, "c_reactive_protein": 80.0,
        },
        {
            "age": 52, "sex": "Female", "bmi": 22.5, "admission_type": "Elective",
            "hypertension": 0, "diabetes": 0, "coronary_disease": 0,
            "chronic_kidney_disease": 0, "infection_suspected": 0, "trauma_suspected": 0,
            "heart_rate": 82, "systolic_bp": 128, "diastolic_bp": 78,
            "respiratory_rate": 18, "oxygen_saturation": 98, "temperature_c": 36.7,
            "hemoglobin": 13.2, "white_blood_cell": 7.2, "platelet": 230.0,
            "creatinine": 70.0, "lactate": 1.2, "c_reactive_protein": 5.0,
        },
    ]

    status, data = api_request(
        base_url,
        f"/api/v1/predict/batch?model_key={model_key}",
        method="POST",
        data={"patients": patients},
    )

    if status == 200 and isinstance(data, dict):
        total = data.get("total_patients", 0)
        high = data.get("high_risk_count", 0)
        med = data.get("medium_risk_count", 0)
        low = data.get("low_risk_count", 0)
        print(f"PASS total={total}, high={high}, med={med}, low={low}")
        return True

    print(f"FAIL status={status}, detail={data.get('detail', data)}")
    return False


def test_docx_report(base_url: str, model_key: str) -> bool:
    """Test DOCX report export."""
    print(f"5. DOCX报告 POST /api/v1/predict/single/report?model_key={model_key} ... ", end="")

    payload = {
        "age": 65,
        "sex": "Male",
        "bmi": 24.0,
        "admission_type": "Emergency",
        "hypertension": 1,
        "diabetes": 0,
        "coronary_disease": 0,
        "chronic_kidney_disease": 0,
        "infection_suspected": 1,
        "trauma_suspected": 0,
        "heart_rate": 105,
        "systolic_bp": 105,
        "diastolic_bp": 65,
        "respiratory_rate": 24,
        "oxygen_saturation": 93,
        "temperature_c": 38.0,
        "hemoglobin": 11.0,
        "white_blood_cell": 13.0,
        "platelet": 180.0,
        "creatinine": 120.0,
        "lactate": 3.0,
        "c_reactive_protein": 80.0,
    }

    status, data = api_request(
        base_url,
        f"/api/v1/predict/single/report?model_key={model_key}",
        method="POST",
        data=payload,
    )

    if status == 200 and isinstance(data, bytes):
        print(f"PASS {len(data)} bytes (DOCX)")
        return True

    print(f"FAIL status={status}, detail={data.get('detail', data) if isinstance(data, dict) else data}")
    return False


def test_invalid_model(base_url: str) -> bool:
    """Test invalid model key error handling."""
    print("6. 无效模型键测试 ... ", end="")
    payload = {"age": 65, "sex": "Male", "bmi": 24.0, "admission_type": "Emergency",
               "hypertension": 1, "diabetes": 0, "coronary_disease": 0,
               "chronic_kidney_disease": 0, "infection_suspected": 1, "trauma_suspected": 0,
               "heart_rate": 105, "systolic_bp": 105, "diastolic_bp": 65,
               "respiratory_rate": 24, "oxygen_saturation": 93, "temperature_c": 38.0,
               "hemoglobin": 11.0, "white_blood_cell": 13.0, "platelet": 180.0,
               "creatinine": 120.0, "lactate": 3.0, "c_reactive_protein": 80.0}

    status, data = api_request(
        base_url,
        "/api/v1/predict/single?model_key=invalid_model",
        method="POST",
        data=payload,
    )

    if status == 400:
        print(f"PASS (正确返回 400 错误)")
        return True

    print(f"FAIL status={status} (expected 400)")
    return False


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("BggDeepLearning FastAPI 端点测试")
    print(f"Base URL: {args.base_url}")
    print(f"Test model: {args.model_key}")
    print("=" * 60)
    print()

    results = []

    # Run all tests
    results.append(("健康检查", test_health(args.base_url)))
    results.append(("模型列表", test_list_models(args.base_url)))
    results.append(("单个预测", test_single_prediction(args.base_url, args.model_key)))
    results.append(("批量预测", test_batch_prediction(args.base_url, args.model_key)))
    results.append(("DOCX报告", test_docx_report(args.base_url, args.model_key)))
    results.append(("错误处理", test_invalid_model(args.base_url)))

    print()
    print("=" * 60)
    print("测试结果汇总")
    print("-" * 60)

    passed = 0
    failed = 0

    for name, ok in results:
        status = "PASS PASS" if ok else "FAIL FAIL"
        print(f"  {name:12s}  {status}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("-" * 60)
    print(f"通过: {passed}/{len(results)}, 失败: {failed}/{len(results)}")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
