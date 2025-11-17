"""
MiniGPT: Implementação de um modelo Transformer simplificado para geração de texto
Versão 2.0 - Adaptado para corpus customizado + salvamento de modelo

Autor base: MR Autoral
Adaptado: ChatGPT
Data: 2025
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================================
# CONFIGURAÇÕES E HIPERPARÂMETROS
# ============================================================================

@dataclass
class ModelConfig:
    """Configurações do modelo MiniGPT"""
    # Hiperparâmetros de treinamento
    batch_size: int = 32
    block_size: int = 256
    max_iters: int = 3000
    eval_interval: int = 300
    learning_rate: float = 3e-4
    eval_iters: int = 50

    # Arquitetura do modelo
    n_embd: int = 128
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.2

    # Configurações do sistema
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 1337

    # Configurações de dados
    train_split: float = 0.9
    # >>> AQUI VOCÊ COLOCA O NOME DO SEU ARQUIVO DE TEXTO <<<
    input_file: str = 'meu_corpus.txt'

    # Arquivo de saída do modelo treinado
    model_out: str = 'minigpt_checkpoint.pt'


class Logger:
    """Classe para logging estruturado"""

    def __init__(self, name: str = "MiniGPT"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)


# ============================================================================
# PREPARAÇÃO E PROCESSAMENTO DE DADOS
# ============================================================================

class TextDataProcessor:
    """Classe responsável pelo processamento de dados de texto"""

    def __init__(self, config: ModelConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.chars = []
        self.vocab_size = 0
        self.stoi = {}
        self.itos = {}
        self._is_initialized = False

    def create_rich_corpus(self) -> None:
        """
        Cria um corpus de fallback caso o arquivo não exista.
        Se você TEM um arquivo de texto próprio, isso não será usado,
        basta criar o arquivo com o nome em config.input_file.
        """
        corpus_text = """A inteligência artificial representa uma das maiores revoluções tecnológicas da história humana. Desde os primeiros algoritmos até os modernos sistemas de aprendizado profundo, a IA tem transformado nossa sociedade de maneiras inimagináveis.

Python é uma linguagem de programação poderosa e versátil que se tornou fundamental no desenvolvimento de aplicações de inteligência artificial. Sua sintaxe clara e bibliotecas robustas como TensorFlow, PyTorch e scikit-learn fazem dela a escolha preferida de cientistas de dados e engenheiros de machine learning.

O futuro da tecnologia está intrinsecamente ligado ao avanço da inteligência artificial. Modelos de linguagem como GPT, BERT e outros transformers estão revolucionando o processamento de linguagem natural. Estes sistemas conseguem compreender, gerar e traduzir texto com uma precisão impressionante.

Machine learning é o coração da inteligência artificial moderna. Algoritmos de aprendizado supervisionado, não supervisionado e por reforço permitem que máquinas aprendam padrões complexos a partir de dados. Redes neurais profundas, especialmente arquiteturas como transformers, têm mostrado capacidades extraordinárias em tarefas como reconhecimento de imagem, processamento de linguagem natural e geração de conteúdo.

A ciência da computação evoluiu dramaticamente nas últimas décadas. Estruturas de dados eficientes, algoritmos otimizados e arquiteturas de sistemas distribuídos são fundamentais para lidar com o volume massivo de dados da era digital. Conceitos como big data, cloud computing e edge computing definem o panorama tecnológico atual.

Deep learning utiliza redes neurais com múltiplas camadas para modelar e compreender dados complexos. Técnicas como convolução, atenção e normalização por lotes têm permitido avanços significativos em visão computacional, processamento de linguagem natural e outras áreas da inteligência artificial.

Transformers revolucionaram o campo do processamento de linguagem natural. O mecanismo de atenção permite que estes modelos capturem dependências de longo alcance em sequências, resultando em melhor compreensão contextual. Modelos como GPT, BERT e T5 demonstram o poder desta arquitetura.

A programação em Python oferece uma sintaxe intuitiva e uma vasta gama de bibliotecas especializadas. NumPy para computação numérica, Pandas para manipulação de dados, Matplotlib para visualização e Jupyter Notebooks para desenvolvimento interativo formam um ecossistema poderoso para ciência de dados.

Algoritmos de otimização como gradiente descendente, Adam e RMSprop são essenciais para treinar modelos de machine learning. A retropropagação permite que redes neurais ajustem seus pesos de forma eficiente, enquanto técnicas de regularização como dropout e normalização por lotes previnem overfitting.

O processamento de linguagem natural combina linguística computacional com machine learning para permitir que computadores compreendam e gerem linguagem humana. Tarefas como análise de sentimento, tradução automática, sumarização de texto e geração de conteúdo são aplicações práticas desta área.

Redes neurais artificiais são inspiradas no funcionamento do cérebro humano. Neurônios artificiais processam informações através de funções de ativação como ReLU, sigmoid e tanh. Arquiteturas como redes convolucionais, recorrentes e transformers são especializadas para diferentes tipos de dados e tarefas.

A era dos dados massivos exige ferramentas e técnicas especializadas. Hadoop, Spark e outras tecnologias de big data permitem processar volumes enormes de informação. Bancos de dados NoSQL como MongoDB e Cassandra oferecem flexibilidade para dados não estruturados.

Aprendizado por reforço permite que agentes artificiais aprendam através da interação com ambientes. Algoritmos como Q-learning, policy gradients e actor-critic têm sido aplicados com sucesso em jogos, robótica e sistemas de recomendação.

A visão computacional utiliza técnicas de deep learning para interpretar e analisar imagens e vídeos. Redes neurais convolucionais são particularmente eficazes para reconhecimento de objetos, detecção facial e segmentação de imagens.

Sistemas distribuídos são fundamentais para aplicações modernas de grande escala. Conceitos como microserviços, containerização com Docker e orquestração com Kubernetes permitem construir aplicações robustas e escaláveis.

A segurança cibernética torna-se cada vez mais crítica à medida que nossa dependência da tecnologia aumenta. Criptografia, autenticação multifator e detecção de anomalias baseada em machine learning são componentes essenciais da proteção digital.

Computação quântica representa a próxima fronteira da computação. Qubits, superposição e emaranhamento quântico prometem resolver problemas computacionalmente intratáveis para computadores clássicos.

A ética em inteligência artificial é uma preocupação crescente. Questões sobre viés algorítmico, privacidade de dados, transparência de modelos e impacto social da automação requerem consideração cuidadosa no desenvolvimento de sistemas de IA.

DevOps integra desenvolvimento e operações para acelerar a entrega de software. Práticas como integração contínua, entrega contínua e infraestrutura como código são essenciais para equipes de desenvolvimento modernas.

A internet das coisas conecta dispositivos físicos à internet, gerando volumes massivos de dados. Sensores, atuadores e sistemas embarcados criam uma rede ubíqua de dispositivos inteligentes.

Blockchain e tecnologias de ledger distribuído oferecem novas possibilidades para sistemas descentralizados. Contratos inteligentes, criptomoedas e aplicações descentralizadas representam inovações significativas em sistemas distribuídos.

A realidade aumentada e virtual estão transformando como interagimos com informação digital. Tecnologias imersivas têm aplicações em educação, entretenimento, treinamento e visualização de dados.

Computação em nuvem democratizou o acesso a recursos computacionais poderosos. Serviços como AWS, Azure e Google Cloud Platform oferecem infraestrutura escalável para aplicações de qualquer tamanho.

A análise de dados revela insights valiosos escondidos em grandes conjuntos de dados. Técnicas estatísticas, visualização de dados e machine learning trabalham juntas para extrair conhecimento acionável.

Arquiteturas de software modernas enfatizam modularidade, escalabilidade e manutenibilidade. Padrões de design, princípios SOLID e arquiteturas hexagonais guiam o desenvolvimento de sistemas robustos.

A automação está transformando indústrias inteiras. Robótica, sistemas autônomos e processos automatizados aumentam eficiência e reduzem custos operacionais.

Interfaces de usuário intuitivas são cruciais para a adoção de tecnologia. Design centrado no usuário, experiência do usuário e acessibilidade garantem que sistemas complexos sejam utilizáveis por todos.

A computação de borda traz processamento mais próximo dos dados, reduzindo latência e melhorando performance. Edge computing é especialmente importante para aplicações em tempo real e IoT.

Sistemas de recomendação utilizam machine learning para personalizar experiências. Filtragem colaborativa, filtragem baseada em conteúdo e deep learning criam recomendações relevantes para usuários.

A ciência de dados combina estatística, programação e conhecimento de domínio para extrair insights de dados. O processo de descoberta de conhecimento em dados envolve coleta, limpeza, análise e interpretação.

Tecnologias emergentes como computação neuromórfica, computação DNA e computação óptica prometem revolucionar a computação nas próximas décadas.

A colaboração entre humanos e inteligência artificial está criando novas possibilidades. Sistemas de IA aumentam capacidades humanas em vez de simplesmente substituí-las, criando parcerias produtivas.

O desenvolvimento sustentável de tecnologia considera impacto ambiental e social. Green computing, eficiência energética e responsabilidade social corporativa são considerações importantes no desenvolvimento tecnológico.

A educação em tecnologia deve evoluir para preparar profissionais para o futuro digital. Pensamento computacional, alfabetização em dados e habilidades de programação tornam-se fundamentais para todas as profissões.

A pesquisa em inteligência artificial continua avançando rapidamente. Novos algoritmos, arquiteturas e técnicas são desenvolvidos constantemente, empurrando os limites do que é possível com sistemas artificiais.

A interdisciplinaridade é essencial na tecnologia moderna. Colaboração entre cientistas da computação, matemáticos, psicólogos, linguistas e especialistas de domínio resulta em soluções mais robustas e inovadoras.

O futuro da humanidade está intimamente ligado ao desenvolvimento responsável da tecnologia. Equilibrar inovação com considerações éticas, sociais e ambientais é crucial para um futuro sustentável e próspero."""
        with open(self.config.input_file, 'w', encoding='utf-8') as f:
            f.write(corpus_text)

        self.logger.info(
            f"Corpus rico criado: '{self.config.input_file}' ({len(corpus_text)} caracteres)"
        )

    def _build_vocabulary(self, text: str) -> None:
        """Constrói o vocabulário a partir do texto"""
        # Cria vocabulário
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Cria mapeamentos
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self._is_initialized = True
        self.logger.info(f"Vocabulário criado com {self.vocab_size} caracteres únicos")
        self.logger.info(
            f"Amostra do vocabulário: "
            f"{''.join(self.chars[:30])}{'...' if len(self.chars) > 30 else ''}"
        )

    def load_and_process_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Carrega e processa os dados de texto"""
        # Verifica se o arquivo existe, senão cria um corpus rico
        if not Path(self.config.input_file).exists():
            self.logger.warning(
                f"Arquivo '{self.config.input_file}' não encontrado. "
                f"Criando corpus de exemplo..."
            )
            self.create_rich_corpus()

        # Carrega o texto
        try:
            with open(self.config.input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            self.logger.error(f"Erro ao carregar arquivo: {e}")
            raise

        if not text.strip():
            raise ValueError("Arquivo de texto está vazio")

        self.logger.info(f"Texto carregado: {len(text)} caracteres")

        # Constrói vocabulário
        self._build_vocabulary(text)

        # Tokeniza o texto
        try:
            encoded = self.encode(text)
            if not isinstance(encoded, list):
                raise RuntimeError("encode() deve retornar uma lista de inteiros")
            data = torch.tensor(encoded, dtype=torch.long)
        except Exception as e:
            self.logger.error(f"Erro na tokenização: {e}")
            raise

        # Divide em treino e validação
        n = int(self.config.train_split * len(data))
        train_data = data[:n]
        val_data = data[n:]

        self.logger.info(f"Dados divididos: {len(train_data)} treino, {len(val_data)} validação")
        return train_data, val_data

    def encode(self, text: str) -> list:
        """Codifica texto em lista de inteiros

        Corrigido: garantir que sempre retorna uma lista (vazia se necessário) e
        logar/ou filtrar caracteres desconhecidos ao invés de lançar exceção que
        deixa a função retornar None.
        """
        if not self._is_initialized:
            raise RuntimeError("Vocabulário não foi inicializado. "
                               "Chame load_and_process_data() primeiro.")

        encoded = []
        missing = set()
        for c in text:
            idx = self.stoi.get(c)
            if idx is None:
                missing.add(c)
            else:
                encoded.append(idx)

        if missing:
            # Mostra apenas uma amostra dos caracteres faltantes para não poluir o log
            sample = ''.join(list(missing)[:50])
            self.logger.warning(
                f"{len(missing)} caracteres não estão no vocabulário e serão ignorados. "
                f"Amostra: '{sample}'"
            )

        return encoded

    def decode(self, tokens: list) -> str:
        """Decodifica lista de inteiros em texto

        Robustez: aceita tensores/ints e filtra tokens desconhecidos.
        """
        if not self._is_initialized:
            raise RuntimeError("Vocabulário não foi inicializado. "
                               "Chame load_and_process_data() primeiro.")

        out_chars = []
        for t in tokens:
            try:
                ti = int(t)
            except Exception:
                self.logger.warning(f"Token inválido durante decode: {t}")
                continue
            ch = self.itos.get(ti)
            if ch is None:
                self.logger.warning(f"Token {ti} não encontrado no vocabulário — ignorando")
                continue
            out_chars.append(ch)
        return ''.join(out_chars)


class DataLoader:
    """Classe para carregamento de lotes de dados"""

    def __init__(
        self,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        config: ModelConfig
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.config = config

        # Validações
        if len(train_data) < config.block_size:
            raise ValueError(
                f"Dados de treino muito pequenos: {len(train_data)} < {config.block_size}"
            )
        if len(val_data) < config.block_size:
            raise ValueError(
                f"Dados de validação muito pequenos: {len(val_data)} < {config.block_size}"
            )

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == 'train' else self.val_data

        # Garante que temos dados suficientes
        max_start_idx = len(data) - self.config.block_size
        if max_start_idx <= 0:
            raise ValueError(
                f"Dados insuficientes para block_size {self.config.block_size}"
            )

        # Índices de início aleatórios: 0 <= i <= max_start_idx-1
        ix = torch.randint(0, max_start_idx, (self.config.batch_size,))
        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.config.block_size+1] for i in ix])

        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y


# ============================================================================
# COMPONENTES DO MODELO TRANSFORMER
# ============================================================================

class AttentionHead(nn.Module):
    """Uma única cabeça do mecanismo de autoatenção"""

    def __init__(self, config: ModelConfig, head_size: int):
        super().__init__()
        self.head_size = head_size
        self.config = config

        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)

        self.register_buffer(
            'tril',
            torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Calcula scores de atenção
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Agregação ponderada
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Múltiplas cabeças de autoatenção em paralelo"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head

        self.heads = nn.ModuleList([
            AttentionHead(config, head_size) for _ in range(config.n_head)
        ])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Rede feed-forward"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Bloco Transformer completo"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ============================================================================
# MODELO PRINCIPAL
# ============================================================================

class MiniGPT(nn.Module):
    """Modelo MiniGPT completo"""

    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(*[
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

        # Inicialização de pesos
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Inicialização de pesos do modelo"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape

        # Validação de entrada
        if T > self.config.block_size:
            raise ValueError(f"Sequência muito longa: {T} > {self.config.block_size}")

        tok_emb = self.token_embedding_table(idx)
        pos_indices = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos_indices)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Gera novos tokens de forma autoregressiva com controle de temperatura"""
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Limita o contexto ao tamanho máximo
                idx_cond = idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx

    def count_parameters(self) -> int:
        """Conta o número total de parâmetros"""
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# TREINADOR DO MODELO
# ============================================================================

class ModelTrainer:
    """Classe responsável pelo treinamento do modelo"""

    def __init__(
        self,
        model: MiniGPT,
        data_loader: DataLoader,
        config: ModelConfig,
        logger: Logger
    ):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.logger = logger

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )

        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'iterations': []
        }

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """Estima a perda nos conjuntos de treino e validação"""
        out = {}
        self.model.eval()

        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                try:
                    X, Y = self.data_loader.get_batch(split)
                    logits, loss = self.model(X, Y)
                    losses[k] = loss.item()
                except Exception as e:
                    self.logger.error(f"Erro na avaliação: {e}")
                    losses[k] = float('inf')
            out[split] = losses.mean().item()

        self.model.train()
        return out

    def train(self) -> Dict[str, Any]:
        """Executa o treinamento do modelo"""
        self.logger.info("Iniciando treinamento...")
        self.logger.info(
            f"Parâmetros do modelo: {self.model.count_parameters()/1e6:.2f}M"
        )
        self.logger.info(f"Dispositivo: {self.config.device}")

        try:
            for iter_num in range(self.config.max_iters):
                # Avaliação periódica
                if iter_num % self.config.eval_interval == 0 or \
                        iter_num == self.config.max_iters - 1:
                    losses = self.estimate_loss()
                    self.logger.info(
                        f"Iteração {iter_num}: "
                        f"perda treino {losses['train']:.4f}, "
                        f"perda validação {losses['val']:.4f}"
                    )

                    # Salva histórico
                    self.training_history['train_loss'].append(losses['train'])
                    self.training_history['val_loss'].append(losses['val'])
                    self.training_history['iterations'].append(iter_num)

                # Passo de treinamento
                try:
                    xb, yb = self.data_loader.get_batch('train')
                    logits, loss = self.model(xb, yb)

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                except Exception as e:
                    self.logger.error(f"Erro no passo de treinamento {iter_num}: {e}")
                    continue

            self.logger.info("Treinamento concluído!")

        except KeyboardInterrupt:
            self.logger.info("Treinamento interrompido pelo usuário")
        except Exception as e:
            self.logger.error(f"Erro durante o treinamento: {e}")
            raise

        return self.training_history


# ============================================================================
# GERADOR DE TEXTO
# ============================================================================

class TextGenerator:
    """Classe para geração de texto"""

    def __init__(
        self,
        model: MiniGPT,
        processor: TextDataProcessor,
        config: ModelConfig,
        logger: Logger
    ):
        self.model = model
        self.processor = processor
        self.config = config
        self.logger = logger

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8
    ) -> str:
        """Gera texto a partir de um prompt"""
        if not prompt:
            prompt = "A"  # Prompt mínimo

        self.logger.info(
            f"Gerando texto com prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'"
        )

        try:
            # Codifica o prompt
            encoded_prompt = self.processor.encode(prompt)
            if not encoded_prompt:
                self.logger.warning("Prompt vazio após codificação, usando 'A'")
                encoded_prompt = self.processor.encode("A")

            # Garante que o contexto não exceda block_size
            if len(encoded_prompt) > self.config.block_size:
                encoded_prompt = encoded_prompt[-self.config.block_size:]

            context = torch.tensor(
                [encoded_prompt],
                dtype=torch.long,
                device=self.config.device
            )

            # Gera tokens
            generated_tokens = self.model.generate(
                context,
                max_new_tokens,
                temperature=temperature
            )[0].tolist()

            # Decodifica para texto
            generated_text = self.processor.decode(generated_tokens)
            return generated_text

        except Exception as e:
            self.logger.error(f"Erro na geração de texto: {e}")
            return f"Erro na geração: {str(e)}"


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal para execução do MiniGPT"""
    try:
        # Configuração
        config = ModelConfig()
        logger = Logger()

        logger.info("Iniciando MiniGPT v2.0...")
        logger.info(
            f"Configurações: batch_size={config.batch_size}, "
            f"block_size={config.block_size}"
        )

        # Configuração de reprodutibilidade
        torch.manual_seed(config.seed)

        # Processamento de dados
        logger.info("Carregando e processando dados...")
        processor = TextDataProcessor(config, logger)
        train_data, val_data = processor.load_and_process_data()
        data_loader = DataLoader(train_data, val_data, config)

        # Criação do modelo
        logger.info("Criando modelo...")
        model = MiniGPT(config, processor.vocab_size)
        model = model.to(config.device)

        # Treinamento
        logger.info("Iniciando treinamento...")
        trainer = ModelTrainer(model, data_loader, config, logger)
        training_history = trainer.train()

        # Geração de texto de exemplo (opcional)
        generator = TextGenerator(model, processor, config, logger)

        prompts = [
            "artificial",
            "Python linguagem",
            "tecnologia",
            "Machine learning",
            "Deep learning",
            "Transformers",
            "computação",
            "otimização"
        ]

        logger.info("\n=== EXEMPLOS DE GERAÇÃO DE TEXTO ===")
        for i, prompt in enumerate(prompts, 1):
            try:
                generated_text = generator.generate_text(
                    prompt,
                    max_new_tokens=200,
                    temperature=0.8
                )

                print(f"\n{'='*80}")
                print(f"EXEMPLO {i}/8")
                print(f"{'='*80}")
                print(f"Prompt: '{prompt}'")
                print(f"{'-'*80}")
                print(f"Texto Gerado:")
                print(generated_text)
                print(f"{'='*80}")

            except Exception as e:
                logger.error(f"Erro ao gerar texto para prompt '{prompt}': {e}")

        # Estatísticas finais
        final_train_loss = training_history['train_loss'][-1] if training_history['train_loss'] else 'N/A'
        final_val_loss = training_history['val_loss'][-1] if training_history['val_loss'] else 'N/A'

        logger.info(f"\n=== ESTATÍSTICAS FINAIS ===")
        logger.info(f"Treinamento concluído com sucesso!")
        logger.info(f"Perda final de treino: {final_train_loss}")
        logger.info(f"Perda final de validação: {final_val_loss}")
        logger.info(f"Parâmetros do modelo: {model.count_parameters()/1e6:.2f}M")
        logger.info(f"Tamanho do vocabulário: {processor.vocab_size}")

        # >>> SALVA O MODELO + CONFIG + VOCAB EM ARQUIVO <<<
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config.__dict__,
            'vocab': {
                'chars': processor.chars,
                'stoi': processor.stoi,
                'itos': processor.itos,
                'vocab_size': processor.vocab_size,
            },
        }
        torch.save(checkpoint, config.model_out)
        logger.info(f"Modelo salvo em: {config.model_out}")

    except Exception as e:
        print(f"Erro fatal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()