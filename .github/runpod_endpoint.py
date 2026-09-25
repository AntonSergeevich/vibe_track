"""Найти или создать serverless-эндпоинт ACE-Step на RunPod.

Эндпоинт создаётся с workersMin=0: без запросов воркеров нет и платить
не за что -- деньги идут только за секунды генерации (и за холодный старт,
когда воркер поднимается). Печатает все ответы API целиком: поля REST API
RunPod проверяются этим же прогоном, а не принимаются на веру.
"""
import os
import sys

import requests

API = "https://rest.runpod.io/v1"
NAME = "vibetrack-acestep"
# Образ со встроенными весами base-чекпойнта: задача lego ("дописать
# партии поверх исходника") есть только в base, в turbo её нет.
IMAGE = "ghcr.io/quang101182/acestep-serverless:latest"
GPUS = ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A4500", "NVIDIA L4"]

if not os.environ.get("RUNPOD_API_KEY"):
    sys.exit("Секрет RUNPOD_API_KEY не задан в настройках репозитория "
             "(Settings -> Secrets and variables -> Actions)")
headers = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}


def call(method, path, **kwargs):
    response = requests.request(method, f"{API}{path}", headers=headers, timeout=60, **kwargs)
    print(f"{method} {path} -> {response.status_code}\n{response.text[:2000]}\n")
    response.raise_for_status()
    return response.json()


def main() -> None:
    endpoint = next((e for e in call("GET", "/endpoints") if e.get("name", "").startswith(NAME)),
                    None)
    if endpoint is None:
        template = next((t for t in call("GET", "/templates") if t.get("name") == NAME), None)
        if template is None:
            template = call("POST", "/templates", json={
                "name": NAME, "imageName": IMAGE, "isServerless": True,
                "containerDiskInGb": 40,
            })
        endpoint = call("POST", "/endpoints", json={
            "name": NAME, "templateId": template["id"], "gpuTypeIds": GPUS,
            "workersMin": 0, "workersMax": 1, "idleTimeout": 5,
            "executionTimeoutMs": 2400000,
        })
    print("ENDPOINT_ID", endpoint["id"])
    with open(os.environ["GITHUB_ENV"], "a") as env:
        env.write(f"ENDPOINT_ID={endpoint['id']}\n")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as error:
        sys.exit(f"RunPod API отказал: {error}")
