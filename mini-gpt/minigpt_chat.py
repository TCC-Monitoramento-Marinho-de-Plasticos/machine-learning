"""
Chat com MiniGPT treinado (usa o checkpoint salvo em minigpt_train.py)
"""

import torch

from minigpt_train import (
    ModelConfig,
    MiniGPT,
    TextDataProcessor,
    TextGenerator,
    Logger,
)


def load_model_from_checkpoint(
    checkpoint_path: str = "minigpt_checkpoint.pt",
    device: str | None = None,
):
    logger = Logger("MiniGPT-Chat")

    # Carrega checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    cfg_dict = ckpt["config"]
    # Se quiser forçar CPU/GPU, pode sobrescrever aqui
    if device is not None:
        cfg_dict["device"] = device

    config = ModelConfig(**cfg_dict)

    # Ajusta device final
    if device is not None:
        config.device = device

    logger.info(f"Carregando modelo em {config.device}...")

    vocab = ckpt["vocab"]

    # Reconstrói o processor com o vocabulário salvo
    processor = TextDataProcessor(config, logger)
    processor.chars = vocab["chars"]
    processor.vocab_size = vocab["vocab_size"]
    processor.stoi = vocab["stoi"]
    processor.itos = vocab["itos"]
    processor._is_initialized = True  # habilita encode/decode

    # Cria modelo e carrega pesos
    model = MiniGPT(config, processor.vocab_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(config.device)
    model.eval()

    return model, processor, config, logger


def chat_loop():
    # Se quiser forçar CPU: device="cpu"
    model, processor, config, logger = load_model_from_checkpoint(device=None)
    generator = TextGenerator(model, processor, config, logger)

    print("\n====================== MINI GPT CHAT ======================")
    print("Digite sua mensagem e pressione Enter.")
    print("Digite 'sair', 'exit' ou 'quit' para encerrar.")
    print("===========================================================\n")

    while True:
        try:
            user_input = input("Você: ")
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando chat...")
            break

        if user_input.strip().lower() in {"sair", "exit", "quit"}:
            print("MiniGPT: Até mais!")
            break

        # Gera resposta
        resposta = generator.generate_text(
            user_input,
            max_new_tokens=300,
            temperature=0.3,  # Pode ajustar: menor = mais “determinístico”
        )

        print("\nMiniGPT:\n" + resposta + "\n" + "-" * 60)


if __name__ == "__main__":
    chat_loop()
