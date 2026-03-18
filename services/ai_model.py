
from dao.entities import AIModelEntity
from dao.repositories import AIModelRepository
from core.database.oracle import get_session

class AIModelService:
    def __init__(self):
        pass

    @property
    def oracle_session(self):
        return get_session()
    
    async def get_model_name_by_id(self, model_id: int) -> str:
        async with self.oracle_session as session:
            repo = AIModelRepository(session)
            model = await repo.get_by_id(model_id)
            return model.model_name