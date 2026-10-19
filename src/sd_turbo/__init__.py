import torch
from diffusers import AutoPipelineForText2Image

MODELO = "stabilityai/sd-turbo"
SALIDA = "imagen.png"


def elegir_dispositivo():
    # cuda - nvidia | mps - Mac con chip M | cpu - cualquier equipo
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    dispositivo = elegir_dispositivo()
    # float16 solo en GPU; en CPU se usa float32 (mayor precision)
    tipo = torch.float16 if dispositivo == "cuda" else torch.float32

    print(f"Cargando modelo en {dispositivo} ...")
    modelo = AutoPipelineForText2Image.from_pretrained(MODELO, torch_dtype=tipo)
    modelo = modelo.to(dispositivo)

    prompt = input("Escribe el prompt de la imagen: ")

    print("Generando la imagen...")
    imagen = modelo(
        prompt=prompt,
        num_inference_steps=4,  # SD-Turbo funciona con 1 a 4 pasos
        guidance_scale=0.0,  # SD-Turbo se entreno sin guidance (ignora negative_prompt)
        height=512,  # resolucion nativa del modelo
        width=512,
    ).images[0]

    imagen.save(SALIDA)
    print(f"Imagen guardada en {SALIDA} !")


if __name__ == "__main__":
    main()
