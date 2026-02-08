# Referência da API

## Endpoint: `/analyze`

Analisa um texto em busca de vieses implícitos.

- **Método**: `POST`
- **URL**: `/analyze`
- **Content-Type**: `application/json`

### Parâmetros

| Nome | Tipo | Obrigatório | Descrição |
|------|------|-------------|-----------|
| `text` | string | Sim | O texto da descrição da vaga a ser analisado. |

### Exemplo de Requisição

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "Procuramos um ninja jovem para nosso time."}'
```

### Exemplo de Resposta (200 OK)

```json
{
  "text": "Procuramos um ninja jovem para nosso time.",
  "scores": {
    "gender": 0.85,
    "age": 0.92,
    "culture": 0.12
  },
  "flags": [
    "gender",
    "age"
  ]
}
```

### Códigos de Retorno

- **200**: Sucesso.
- **422**: Erro de validação (ex: campo `text` ausente).
- **500**: Erro interno do servidor.
