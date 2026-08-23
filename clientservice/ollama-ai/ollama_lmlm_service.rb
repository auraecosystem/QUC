# frozen_string_literal: true

require 'ollama-ai'

class OllamaLMLMService
  attr_reader :client, :default_model

  def initialize(
    address: 'http://localhost:11434',
    bearer_token: ENV.fetch('OLLAMA_BEARER_TOKEN', nil),
    default_model: 'llama3'
  )
    @client = Ollama.new(
      credentials: {
        address: address,
        bearer_token: bearer_token
      },
      options: { server_sent_events: true }
    )
    @default_model = default_model
  end

  # Executes conversational chat requests with optional SSE streaming
  def chat(messages:, model: nil, &block)
    model_to_use = model || @default_model
    @client.chat(
      {
        model: model_to_use,
        messages: messages
      },
      &block
    )
  rescue Ollama::Errors::OllamaError => e
    { error: e.class.name, message: e.message, status: e.try(:response)&.status }
  end

  # Generates dense float vector embeddings for input text
  def generate_embedding(prompt:, model: nil)
    model_to_use = model || @default_model
    response = @client.embeddings({ model: model_to_use, prompt: prompt })
    response.first&.fetch('embedding', [])
  rescue Ollama::Errors::OllamaError => e
    warn "Embedding generation failed: #{e.message}"
    []
  end

  # Creates a localized model variant with custom system prompts
  def create_localized_model(name:, base_model: 'llama3', system_prompt:)
    modelfile = "FROM #{base_model}\nSYSTEM #{system_prompt}"
    @client.create({ name: name, modelfile: modelfile })
  end

  # Copies and pushes a model namespace to the Ollama registry
  def publish_model(source_name:, destination_namespace:)
    @client.copy({ source: source_name, destination: destination_namespace })
    @client.push({ name: destination_namespace }) do |event, _raw|
      puts "Push Status: #{event['status']}" if event['status']
    end
  end
end

# --- Usage Example ---
if __FILE__ == $PROGRAM_NAME
  service = OllamaLMLMService.new(default_model: 'llama3')

  # Chat request with streaming SSE block
  service.chat(
    messages: [{ role: 'user', content: 'Explain quantum computing in one sentence.' }]
  ) do |event, _raw|
    print event.dig('message', 'content')
  end
end


