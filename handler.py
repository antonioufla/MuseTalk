import runpod
import os
import time
import requests
import subprocess
import shutil
import uuid
from pathlib import Path

# Paths
WORKSPACE = Path("/workspace")
MODEL_DIR = WORKSPACE / "models"
OUTPUT_DIR = WORKSPACE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configs do modelo
UNET_MODEL_PATH = str(MODEL_DIR / "musetalkV15" / "unet.pth")
UNET_CONFIG = str(MODEL_DIR / "musetalkV15" / "musetalk.json")


def download_file(url: str, dest: Path) -> Path:
    """Baixa arquivo de URL para destino."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(dest, "wb") as f:
        f.write(response.content)
    return dest


def run_inference(image_path: str, audio_path: str, output_dir: str, job_id: str) -> str:
    """Executa o MuseTalk."""
    result_dir = os.path.join(output_dir, job_id)
    os.makedirs(result_dir, exist_ok=True)

    # inference.py NÃO aceita --video_path/--audio_path na CLI; lê de YAML via --inference_config
    config_path = os.path.join(result_dir, "inference_config.yaml")
    with open(config_path, "w") as f:
        f.write(f"task_0:\n  video_path: \"{image_path}\"\n  audio_path: \"{audio_path}\"\n")

    cmd = [
        "python3", "-m", "scripts.inference",
        "--inference_config", config_path,
        "--result_dir", result_dir,
        "--unet_model_path", UNET_MODEL_PATH,
        "--unet_config", UNET_CONFIG,
        "--version", "v15",
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(WORKSPACE)
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Inference failed:\n{proc.stderr}")

    # Localizar vídeo gerado
    output_files = list(Path(result_dir).glob("*.mp4"))
    if not output_files:
        raise RuntimeError(f"Nenhum vídeo gerado em {result_dir}")

    return str(output_files[0])


def handler(job):
    """Handler principal do RunPod."""
    print("[MuseTalk] handler v2 - YAML config fix ativo (2026-03-04)")
    job_input = job.get("input", {})

    # Obter URLs
    image_url = job_input.get("image_url")
    audio_url = job_input.get("audio_url")

    if not image_url:
        return {"error": "image_url é obrigatório"}
    if not audio_url:
        return {"error": "audio_url é obrigatório"}

    job_id = job.get("id", str(uuid.uuid4()))
    tmp_dir = OUTPUT_DIR / job_id
    tmp_dir.mkdir(exist_ok=True)

    try:
        # Baixar arquivos
        print(f"[MuseTalk] Baixando imagem: {image_url}")
        image_path = download_file(image_url, tmp_dir / "input.png")

        print(f"[MuseTalk] Baixando áudio: {audio_url}")
        audio_path = download_file(audio_url, tmp_dir / "input.wav")

        # Executar inferência
        print(f"[MuseTalk] Iniciando inferência...")
        start_time = time.time()

        output_video = run_inference(
            str(image_path),
            str(audio_path),
            str(OUTPUT_DIR),
            job_id
        )

        elapsed = round(time.time() - start_time, 2)
        print(f"[MuseTalk] Concluído em {elapsed}s")

        # Ler vídeo como bytes base64
        import base64
        with open(output_video, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "status": "success",
            "model": "MuseTalk",
            "execution_time_seconds": elapsed,
            "video_base64": video_b64,
            "output_path": output_video
        }

    except Exception as e:
        return {"error": str(e), "model": "MuseTalk"}

    finally:
        # Limpar arquivos temporários de entrada
        if (tmp_dir / "input.png").exists():
            (tmp_dir / "input.png").unlink()
        if (tmp_dir / "input.wav").exists():
            (tmp_dir / "input.wav").unlink()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
