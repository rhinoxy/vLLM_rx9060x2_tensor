# AnythingLLM

All-in-one AI アプリケーション **AnythingLLM** (`mintplexlabs/anythingllm`) の環境です。

## 起動・停止方法

ディレクトリ: `/home/k-takeda/gAI-LLM/anything-llm`

- **起動**:
  ```bash
  ./start.sh
  # または
  /home/k-takeda/gAI-LLM/bin/docker-compose up -d
  ```

- **停止**:
  ```bash
  ./stop.sh
  # または
  /home/k-takeda/gAI-LLM/bin/docker-compose down
  ```

- **ログ確認**:
  ```bash
  ./logs.sh
  # または
  /home/k-takeda/gAI-LLM/bin/docker-compose logs -f
  ```

---

## アクセス先

- **Web UI**: [http://localhost:3001](http://localhost:3001)

---

## Ollama 連携設定

AnythingLLM はホストネットワークモード（`network_mode: host`）で動作しているため、ホスト上の Ollama に直接アクセスできます。

- **Ollama Base URL**: `http://127.0.0.1:11434` または `http://localhost:11434`
- 設定画面（Settings → AI Providers → LLM）で **Ollama** を選択し、上記URLを入力するとダウンロード済みのモデル（`qwen3.8:27b`, `gemma4:26B` など）が自動検出されます。
- Embedding Engine でも同様に **Ollama** を選択し、`bge-m3:latest` を指定できます。
