# Essential Nodes - Technical Specification

This document provides detailed technical specifications for the most essential missing nodes identified in the NODE_ANALYSIS.md.

## 1. Database Integration Nodes

### PostgreSQL Nodes

#### PostgreSQLConnect
```python
class PostgreSQLConnect(BaseNode):
    """
    Establish a connection to a PostgreSQL database.
    
    Use cases:
    - Connect to application databases
    - Data warehousing connections
    - Analytics database access
    
    Tags: database, postgresql, sql, connection
    """
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(description="Database name")
    username: str = Field(description="Database username")
    password: str = Field(description="Database password", secret=True)
    ssl_mode: str = Field(default="prefer", description="SSL mode")
    
    async def process(self, context: ProcessingContext) -> DBConnection:
        # Returns a connection handle
        pass
```

#### PostgreSQLQuery
```python
class PostgreSQLQuery(BaseNode):
    """
    Execute a SELECT query on a PostgreSQL database.
    
    Returns results as a list of dictionaries.
    
    Use cases:
    - Fetch data for analysis
    - Read configuration data
    - Query business data
    
    Tags: database, postgresql, sql, query, select
    """
    connection: DBConnection = Field(description="Database connection")
    query: str = Field(description="SQL SELECT query")
    parameters: dict[str, Any] = Field(default={}, description="Query parameters")
    
    async def process(self, context: ProcessingContext) -> list[dict[str, Any]]:
        # Execute query and return results
        pass
```

#### PostgreSQLExecute
```python
class PostgreSQLExecute(BaseNode):
    """
    Execute an INSERT, UPDATE, DELETE or DDL statement.
    
    Use cases:
    - Insert data into tables
    - Update records
    - Delete data
    - Create/alter tables
    
    Tags: database, postgresql, sql, execute, insert, update, delete
    """
    connection: DBConnection = Field(description="Database connection")
    statement: str = Field(description="SQL statement")
    parameters: dict[str, Any] = Field(default={}, description="Statement parameters")
    
    async def process(self, context: ProcessingContext) -> int:
        # Returns number of affected rows
        pass
```

### MongoDB Nodes

#### MongoDBConnect
```python
class MongoDBConnect(BaseNode):
    """
    Connect to a MongoDB database.
    
    Use cases:
    - NoSQL database access
    - Document storage
    - Real-time applications
    
    Tags: database, mongodb, nosql, connection
    """
    connection_string: str = Field(description="MongoDB connection string", secret=True)
    database: str = Field(description="Database name")
    
    async def process(self, context: ProcessingContext) -> MongoDBConnection:
        pass
```

#### MongoDBFind
```python
class MongoDBFind(BaseNode):
    """
    Find documents in a MongoDB collection.
    
    Use cases:
    - Query document stores
    - Retrieve application data
    - Search JSON documents
    
    Tags: database, mongodb, query, find, nosql
    """
    connection: MongoDBConnection = Field(description="MongoDB connection")
    collection: str = Field(description="Collection name")
    filter: dict[str, Any] = Field(default={}, description="Query filter")
    projection: dict[str, Any] = Field(default={}, description="Field projection")
    limit: int = Field(default=0, description="Limit results (0 = no limit)")
    
    async def process(self, context: ProcessingContext) -> list[dict[str, Any]]:
        pass
```

## 2. Cloud Storage Nodes

### AWS S3 Nodes

#### S3Upload
```python
class S3Upload(BaseNode):
    """
    Upload a file or data to AWS S3.
    
    Use cases:
    - Store generated files in cloud
    - Backup data to S3
    - Share files via S3 URLs
    
    Tags: aws, s3, cloud, storage, upload
    """
    bucket: str = Field(description="S3 bucket name")
    key: str = Field(description="Object key (path)")
    data: Union[bytes, str, AssetRef] = Field(description="Data to upload")
    content_type: str = Field(default="", description="Content type")
    access_key: str = Field(default="", description="AWS access key", secret=True)
    secret_key: str = Field(default="", description="AWS secret key", secret=True)
    region: str = Field(default="us-east-1", description="AWS region")
    
    async def process(self, context: ProcessingContext) -> str:
        # Returns S3 URL
        pass
```

#### S3Download
```python
class S3Download(BaseNode):
    """
    Download a file from AWS S3.
    
    Use cases:
    - Retrieve stored files
    - Load data from S3
    - Access cloud backups
    
    Tags: aws, s3, cloud, storage, download
    """
    bucket: str = Field(description="S3 bucket name")
    key: str = Field(description="Object key (path)")
    access_key: str = Field(default="", description="AWS access key", secret=True)
    secret_key: str = Field(default="", description="AWS secret key", secret=True)
    region: str = Field(default="us-east-1", description="AWS region")
    
    async def process(self, context: ProcessingContext) -> AssetRef:
        # Returns downloaded file as asset
        pass
```

#### S3List
```python
class S3List(BaseNode):
    """
    List objects in an S3 bucket.
    
    Use cases:
    - Browse bucket contents
    - Find files by prefix
    - Inventory S3 objects
    
    Tags: aws, s3, cloud, storage, list
    """
    bucket: str = Field(description="S3 bucket name")
    prefix: str = Field(default="", description="Filter by prefix")
    max_keys: int = Field(default=1000, description="Maximum objects to return")
    access_key: str = Field(default="", description="AWS access key", secret=True)
    secret_key: str = Field(default="", description="AWS secret key", secret=True)
    region: str = Field(default="us-east-1", description="AWS region")
    
    async def process(self, context: ProcessingContext) -> list[dict[str, Any]]:
        # Returns list of object metadata
        pass
```

## 3. Anthropic Claude Integration

#### ClaudeChat
```python
class ClaudeChat(BaseNode):
    """
    Generate text responses using Anthropic's Claude models.
    
    Supports Claude 3 family (Opus, Sonnet, Haiku) with context caching,
    vision capabilities, and extended context windows up to 200K tokens.
    
    Use cases:
    - Conversational AI applications
    - Long document analysis
    - Code generation and review
    - Creative writing assistance
    
    Tags: anthropic, claude, llm, chat, ai, text-generation
    """
    prompt: str = Field(description="The user prompt")
    system: str = Field(default="", description="System prompt")
    model: ClaudeModel = Field(default="claude-3-5-sonnet-20241022", description="Claude model")
    max_tokens: int = Field(default=1024, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, ge=0.0, le=1.0, description="Sampling temperature")
    image: ImageRef = Field(default=None, description="Optional image for vision")
    
    async def process(self, context: ProcessingContext) -> str:
        pass
```

#### ClaudeVision
```python
class ClaudeVision(BaseNode):
    """
    Analyze images using Claude's vision capabilities.
    
    Use cases:
    - Image description and analysis
    - OCR and text extraction
    - Visual question answering
    - Chart and diagram interpretation
    
    Tags: anthropic, claude, vision, image-analysis, ai, ocr
    """
    image: ImageRef = Field(description="Image to analyze")
    prompt: str = Field(description="Question or instruction about the image")
    model: ClaudeModel = Field(default="claude-3-5-sonnet-20241022", description="Claude model")
    max_tokens: int = Field(default=1024, description="Maximum tokens to generate")
    
    async def process(self, context: ProcessingContext) -> str:
        pass
```

## 4. Hugging Face Integration

#### HFInference
```python
class HFInference(BaseNode):
    """
    Run inference using Hugging Face Inference API.
    
    Access thousands of open-source models for text, image, audio tasks.
    
    Use cases:
    - Text generation with open models
    - Custom model inference
    - Cost-effective AI processing
    
    Tags: huggingface, inference, ai, ml, open-source
    """
    model_id: str = Field(description="Hugging Face model ID")
    inputs: Union[str, dict] = Field(description="Model inputs")
    parameters: dict = Field(default={}, description="Model parameters")
    api_token: str = Field(default="", description="HF API token", secret=True)
    
    async def process(self, context: ProcessingContext) -> Any:
        pass
```

#### HFTextGeneration
```python
class HFTextGeneration(BaseNode):
    """
    Generate text using Hugging Face models.
    
    Use cases:
    - Open-source LLM inference
    - Custom fine-tuned models
    - Cost-effective text generation
    
    Tags: huggingface, text-generation, llm, ai
    """
    model_id: str = Field(default="mistralai/Mistral-7B-Instruct-v0.2", description="Model ID")
    prompt: str = Field(description="Text prompt")
    max_new_tokens: int = Field(default=256, description="Maximum new tokens")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Nucleus sampling")
    api_token: str = Field(default="", description="HF API token", secret=True)
    
    async def process(self, context: ProcessingContext) -> str:
        pass
```

## 5. Ollama Integration

#### OllamaGenerate
```python
class OllamaGenerate(BaseNode):
    """
    Generate text using local Ollama models.
    
    Run LLMs locally for privacy, cost savings, and offline capability.
    Supports Llama 2/3, Mistral, CodeLlama, and many other models.
    
    Use cases:
    - Private/offline AI processing
    - Cost-free inference
    - Low-latency local generation
    - Development and testing
    
    Tags: ollama, llm, local, ai, text-generation, privacy
    """
    model: str = Field(default="llama2", description="Ollama model name")
    prompt: str = Field(description="Text prompt")
    system: str = Field(default="", description="System prompt")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=512, description="Maximum tokens")
    host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    
    async def process(self, context: ProcessingContext) -> str:
        pass
```

#### OllamaChat
```python
class OllamaChat(BaseNode):
    """
    Chat with local Ollama models using conversation history.
    
    Use cases:
    - Local chatbot applications
    - Multi-turn conversations
    - Context-aware interactions
    
    Tags: ollama, chat, local, ai, conversation
    """
    model: str = Field(default="llama2", description="Ollama model name")
    messages: list[dict[str, str]] = Field(description="Conversation messages")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Sampling temperature")
    host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    
    async def process(self, context: ProcessingContext) -> str:
        pass
```

#### OllamaEmbedding
```python
class OllamaEmbedding(BaseNode):
    """
    Generate embeddings using local Ollama models.
    
    Use cases:
    - Local semantic search
    - Private document indexing
    - Offline vector generation
    
    Tags: ollama, embedding, vector, local, semantic-search
    """
    model: str = Field(default="nomic-embed-text", description="Ollama embedding model")
    text: str = Field(description="Text to embed")
    host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    
    async def process(self, context: ProcessingContext) -> list[float]:
        pass
```

## 6. Pandas Integration

#### DataFrameCreate
```python
class DataFrameCreate(BaseNode):
    """
    Create a pandas DataFrame from various data sources.
    
    Use cases:
    - Convert lists/dicts to DataFrames
    - Create structured data tables
    - Initialize data analysis workflows
    
    Tags: pandas, dataframe, data-processing, create
    """
    data: Union[list[dict], dict[str, list]] = Field(description="Data to create DataFrame from")
    columns: list[str] = Field(default=[], description="Column names")
    
    async def process(self, context: ProcessingContext) -> DataFrameRef:
        pass
```

#### DataFrameFilter
```python
class DataFrameFilter(BaseNode):
    """
    Filter rows in a DataFrame based on conditions.
    
    Use cases:
    - Select subsets of data
    - Apply business rules
    - Data cleaning
    
    Tags: pandas, dataframe, filter, query
    """
    dataframe: DataFrameRef = Field(description="Input DataFrame")
    query: str = Field(description="Pandas query expression")
    
    async def process(self, context: ProcessingContext) -> DataFrameRef:
        pass
```

#### DataFrameGroupBy
```python
class DataFrameGroupBy(BaseNode):
    """
    Group DataFrame rows and compute aggregations.
    
    Use cases:
    - Calculate summary statistics
    - Business intelligence reports
    - Data aggregation pipelines
    
    Tags: pandas, dataframe, groupby, aggregate, statistics
    """
    dataframe: DataFrameRef = Field(description="Input DataFrame")
    group_by: list[str] = Field(description="Columns to group by")
    aggregations: dict[str, str] = Field(description="Column -> aggregation function mapping")
    
    async def process(self, context: ProcessingContext) -> DataFrameRef:
        pass
```

## 7. REST API Integration

#### APIRequest
```python
class APIRequest(BaseNode):
    """
    Make HTTP requests to REST APIs with comprehensive configuration.
    
    Supports all HTTP methods, authentication types, custom headers,
    request/response body handling, and error management.
    
    Use cases:
    - Integrate with external APIs
    - Webhook calls
    - Microservice communication
    - Third-party service integration
    
    Tags: api, rest, http, request, integration
    """
    url: str = Field(description="API endpoint URL")
    method: HTTPMethod = Field(default="GET", description="HTTP method")
    headers: dict[str, str] = Field(default={}, description="Request headers")
    params: dict[str, str] = Field(default={}, description="URL query parameters")
    body: Union[str, dict, None] = Field(default=None, description="Request body")
    auth_type: AuthType = Field(default="none", description="Authentication type")
    auth_token: str = Field(default="", description="Auth token/key", secret=True)
    timeout: int = Field(default=30, description="Request timeout in seconds")
    
    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        # Returns {status, headers, body, json}
        pass
```

#### GraphQLQuery
```python
class GraphQLQuery(BaseNode):
    """
    Execute GraphQL queries against a GraphQL API.
    
    Use cases:
    - Query GraphQL APIs
    - Fetch specific data fields
    - Efficient API data retrieval
    
    Tags: graphql, api, query, integration
    """
    endpoint: str = Field(description="GraphQL endpoint URL")
    query: str = Field(description="GraphQL query")
    variables: dict[str, Any] = Field(default={}, description="Query variables")
    headers: dict[str, str] = Field(default={}, description="Request headers")
    auth_token: str = Field(default="", description="Auth token", secret=True)
    
    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        pass
```

## 8. Advanced File Format Support

#### ParquetRead
```python
class ParquetRead(BaseNode):
    """
    Read data from Apache Parquet files.
    
    Parquet is a columnar storage format optimized for analytics.
    
    Use cases:
    - Load big data analytics files
    - Read data from data lakes
    - Import ML training datasets
    
    Tags: parquet, data, file, read, analytics
    """
    file_path: str = Field(description="Path to parquet file")
    columns: list[str] = Field(default=[], description="Columns to read (empty = all)")
    
    async def process(self, context: ProcessingContext) -> DataFrameRef:
        pass
```

#### ParquetWrite
```python
class ParquetWrite(BaseNode):
    """
    Write data to Apache Parquet format.
    
    Use cases:
    - Export analytics data
    - Store ML datasets efficiently
    - Data lake integration
    
    Tags: parquet, data, file, write, export
    """
    dataframe: DataFrameRef = Field(description="Data to write")
    file_path: str = Field(description="Output file path")
    compression: str = Field(default="snappy", description="Compression codec")
    
    async def process(self, context: ProcessingContext) -> str:
        pass
```

#### YAMLRead
```python
class YAMLRead(BaseNode):
    """
    Parse YAML configuration files.
    
    Use cases:
    - Load configuration files
    - Read Kubernetes manifests
    - Parse CI/CD configs
    
    Tags: yaml, config, parse, read
    """
    file_path: str = Field(description="Path to YAML file")
    
    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        pass
```

#### XMLParse
```python
class XMLParse(BaseNode):
    """
    Parse XML documents into structured data.
    
    Use cases:
    - Parse XML APIs
    - Read RSS/Atom feeds
    - Process SOAP responses
    - Parse XML configuration
    
    Tags: xml, parse, read, data
    """
    xml_string: str = Field(description="XML content to parse")
    
    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        pass
```

## Implementation Priorities

### Immediate (Week 1-2)
1. PostgreSQL integration (most requested)
2. S3 integration (cloud storage foundation)
3. APIRequest node (API integration essential)

### Short-term (Week 3-6)
4. Anthropic Claude integration
5. Ollama integration (local LLM support)
6. Pandas DataFrameCreate, Filter, GroupBy

### Medium-term (Week 7-12)
7. MongoDB integration
8. Hugging Face integration
9. Parquet/YAML/XML support
10. Additional Pandas operations

## Dependencies

### Required Python Packages
- `psycopg2-binary` or `asyncpg` - PostgreSQL
- `pymongo` - MongoDB
- `boto3` - AWS S3
- `anthropic` - Claude API
- `httpx` - HTTP client
- `pandas` - Data processing
- `pyarrow` - Parquet support
- `pyyaml` - YAML support
- `lxml` - XML parsing

### Optional Dependencies
- `sqlalchemy` - Database abstraction
- `google-cloud-storage` - GCS support
- `azure-storage-blob` - Azure support

## Testing Requirements

Each node requires:
1. Unit tests with mocked backends
2. Integration tests (conditional on service availability)
3. Error handling tests
4. Documentation examples
5. Type safety validation

## Security Considerations

- All credentials must be marked as `secret=True`
- Support environment variable configuration
- Implement rate limiting for API calls
- Validate inputs to prevent injection attacks
- Use parameterized queries for SQL
- Implement timeout and retry logic
- Support connection pooling where applicable

## Documentation Requirements

Each node needs:
- Clear docstring with use cases
- Parameter descriptions
- Usage examples
- Error scenarios
- Performance considerations
- Security notes

## Example Usage Patterns

### Database Query Pipeline
```
PostgreSQLConnect -> PostgreSQLQuery -> DataFrameCreate -> DataFrameFilter -> DataFrameGroupBy
```

### API to Cloud Storage
```
APIRequest -> S3Upload -> (downstream processing)
```

### AI Analysis Pipeline
```
S3Download -> ClaudeVision -> MongoDBInsert
```

### Local AI Processing
```
ReadTextFile -> OllamaEmbedding -> ChromaIndexEmbedding -> OllamaGenerate
```

## Migration Path

For teams already using nodetool-base:
1. Introduce nodes with clear naming
2. Provide migration guides
3. Maintain backward compatibility
4. Use deprecation warnings if replacing existing nodes
5. Offer example conversions

## Performance Optimization

- Implement connection pooling
- Cache database connections
- Stream large file operations
- Use async/await throughout
- Implement batch operations where possible
- Add timeout controls
- Support pagination for large datasets
