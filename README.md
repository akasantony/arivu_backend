# Arivu Learning Platform

Arivu is an AI-powered educational platform that adapts to the learning pace of each student. The platform features intelligent agents specialized in different subjects, providing personalized learning experiences based on student grade levels.

## 🚀 Key Features

- **Intelligent Subject Agents**: Specialized AI tutors for Mathematics, Science, Language, and History
- **Adaptive Learning**: Agents adjust teaching styles based on student learning patterns
- **Tool Integration**: Agents can leverage tools like calculators, graph generators, and web search
- **Class-Based Access Control**: Students only see agents appropriate for their grade level
- **Administrative Dashboard**: Manage users, classes, and agent configurations

## 🏗️ Architecture

### Backend (Python/FastAPI/Poetry)

The backend is built with FastAPI and uses Poetry for dependency management. It implements:

- **LangChain Framework**: For creating and managing the educational agents
- **SQLAlchemy ORM**: For database interactions
- **Alembic**: For database migrations
- **JWT Authentication**: For secure user sessions
- **Role-Based Access Control**: For different permission levels

### Frontend (Flutter)

The frontend is built with Flutter for cross-platform support:

- **Responsive UI**: Works on mobile, web, and desktop
- **State Management**: Using provider pattern
- **API Services**: For communication with the backend
- **Real-time Chat**: For interactive learning sessions
- **Educational Visualizations**: For enhanced learning

## 📋 Project Structure

```
arivu/
├── backend/                # Poetry-based Python backend
│   ├── pyproject.toml      # Poetry configuration
│   ├── arivu/              # Main package
│   │   ├── main.py         # FastAPI entry point
│   │   ├── agents/         # LLM Agents
│   │   ├── tools/          # Agent tools
│   │   ├── llm/            # LLM services
│   │   └── ...
│   └── ...
├── frontend/              # Flutter frontend
│   ├── arivu_app/         # Flutter app
│   │   ├── lib/           # Dart code
│   │   ├── assets/        # Static assets
│   │   └── ...
│   └── ...
└── ...
```

## 🛠️ Setup & Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Flutter SDK
- PostgreSQL (for local development)

### Environment Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/arivu.git
   cd arivu
   ```

2. Set up environment variables:

   ```bash
   cp backend/.env.example backend/.env
   # Edit .env with your configuration
   ```

3. Start the services with Docker Compose:
   ```bash
   docker-compose up -d
   ```

### Local Development Setup

#### Backend (Poetry)

```bash
cd backend
poetry install
poetry run alembic upgrade head
poetry run uvicorn arivu.main:app --reload
```

#### Frontend (Flutter)

```bash
cd frontend/arivu_app
flutter pub get
flutter run -d chrome  # For web development
```

## 🧪 Testing

### Backend Tests

```bash
cd backend
poetry run pytest
```

### Frontend Tests

```bash
cd frontend/arivu_app
flutter test
```

## 📚 API Documentation

Once the server is running, API documentation is available at:

- Swagger UI: http://localhost:8000/api/v1/docs
- ReDoc: http://localhost:8000/api/v1/redoc

## 🏛️ Database Schema

The platform uses the following main entities:

- **Users**: Students and administrators
- **Classes**: Grade levels and course groupings
- **Agents**: AI tutors specialized in different subjects
- **Chat Sessions**: Conversations between students and agents

## 👥 User Roles

- **Students**: Can interact with agents assigned to their class
- **Administrators**: Can manage users, classes, and agents

## 🔄 Development Workflow

1. Create new feature branch
2. Implement changes
3. Write tests
4. Submit pull request
5. Review and merge

## 🛡️ Security Considerations

- All API endpoints are protected with JWT authentication
- User passwords are hashed with bcrypt
- Environment variables are used for sensitive configuration
- CORS protection is enabled

## 📈 Roadmap

- [ ] Add support for more subjects
- [ ] Implement progress tracking and analytics
- [ ] Add peer learning features
- [ ] Enable content creation by teachers
- [ ] Implement gamification elements

## 📄 License

[Your License Here]

## 🙏 Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [Flutter](https://flutter.dev/)
- [LangChain](https://langchain.readthedocs.io/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [Poetry](https://python-poetry.org/)
