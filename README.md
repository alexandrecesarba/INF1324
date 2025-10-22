# Pipeline RAG para PDFs

Este repositório contém um pipeline mínimo, inspirado no artigo da Towards AI, para executar sumarização e perguntas e respostas sobre PDFs sem depender de frameworks específicos de RAG. O foco é educacional: os módulos em `main.py` podem ser reaproveitados em apresentações ou integrados em outros projetos.

## Pré-requisitos

- Python 3.10 ou superior.
- Ambiente virtual recomendado (`python -m venv .venv && source .venv/bin/activate`).
- Dependências listadas em [`requirements.txt`](requirements.txt).

Instale os pacotes com:

```bash
pip install -r requirements.txt
```

Alguns modelos do `sentence-transformers` podem realizar download inicial na primeira execução. Garanta acesso à internet ou faça o cache prévio conforme a política da sua infraestrutura.

## Estrutura do código

Todos os utilitários estão centralizados em [`main.py`](main.py):

- `extract_pdf_chunks` lê um PDF com PyMuPDF, gera chunks de texto (e opcionalmente imagens) e retorna metadados prontos para indexação.
- `summarize_chunks` recebe uma função `llm_complete(system_prompt, user_prompt)` e produz resumos por chunk, além de um resumo final.
- `VectorStore` encapsula uma coleção persistente do ChromaDB usando embeddings locais `all-MiniLM-L6-v2`.
- `answer` realiza uma etapa de Q&A baseada nos chunks recuperados.
- `run_demo` costura o fluxo completo e retorna um `DemoResult` com dados úteis para logs ou slides.

## Como executar um exemplo

1. **Defina uma função `llm_complete`:** Ela deve aceitar dois argumentos (prompt do sistema e do usuário) e devolver uma string com a resposta do LLM. Exemplo de stub usando a API da OpenAI:

    ```python
    from openai import OpenAI

    client = OpenAI()

    def llm_complete(system_prompt: str, user_prompt: str) -> str:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content
    ```

    > Substitua pelo modelo de sua preferência (local ou em nuvem). Ajuste autenticação e limites conforme necessário.

2. **Execute o fluxo demonstrativo:**

    ```python
    from main import run_demo

    resultado = run_demo("paper.pdf", llm_complete)

    print("Chunks gerados:", resultado.chunk_count)
    print("Resumo final:\n", resultado.final_summary)
    print("Resposta à pergunta padrão:\n", resultado.answer)
    ```

    O demo usa uma pergunta fixa ("Quais são os objetivos do trabalho?"). Modifique conforme o seu caso.

3. **Rodando consultas personalizadas:**

    Caso deseje controlar cada etapa manualmente:

    ```python
    from main import extract_pdf_chunks, summarize_chunks, VectorStore, answer

    with open("paper.pdf", "rb") as fp:
        dados_pdf = fp.read()

    chunks = extract_pdf_chunks(dados_pdf, max_len=512)
    resumos, resumo_final = summarize_chunks(chunks, llm_complete, mode="brief")

    vetor_store = VectorStore(persist_dir="./chroma", collection="papers_demo")
    vetor_store.add_chunks(chunks)

    resposta, recuperacao = answer("Qual é a metodologia principal?", vetor_store, llm_complete)
    print(resposta)
    ```

    O retorno `recuperacao` inclui documentos, metadados e distâncias para auditoria.

## Dicas para apresentações

- Registre o número de chunks gerados e exemplos de metadados (`source_page`, `image_index`) para ilustrar a extração.
- Mostre o resumo final e como ele é composto a partir dos resumos parciais.
- Logue as distâncias retornadas pelo ChromaDB para discutir relevância e ajuste do `k`.

## Limpeza e persistência

O ChromaDB é inicializado em modo persistente (pasta `./chroma`). Para reiniciar do zero, remova o diretório manualmente:

```bash
rm -rf chroma
```

Tenha cuidado para não apagar dados importantes em produção.

## Licença

Este projeto é fornecido “como está” para fins educativos. Ajuste e reutilize conforme as políticas da sua organização.
