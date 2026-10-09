from enum import Enum, auto


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    
class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        
        if len(prompt_token_ids) == 0:
            raise ValueError("prompt_token_ids is empty, can't generate tokens")
        
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids.copy()
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.generated_token_ids = []
        self.reason = None
        self.status = RequestStatus.WAITING

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.generated_token_ids

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED

    def start(self) -> None:
        if self.status != RequestStatus.WAITING:
            raise RuntimeError("Can't start this request because the request status is not WAITING")
        self.status = RequestStatus.RUNNING

    def append_token(self, token_id: int) -> None:
        if self.status != RequestStatus.RUNNING:
            raise RuntimeError("This request is not in RUNNING state")
        self.generated_token_ids.append(token_id)
        
        if token_id == self.eos_token_id:
            self.reason = "eos"
            self.status = RequestStatus.FINISHED
        elif len(self.generated_token_ids) == self.max_new_tokens:
            self.reason = "length"
            self.status = RequestStatus.FINISHED
            
def main():
    request = Request(
        request_id="example",
        prompt_token_ids=[10, 20],
        max_new_tokens=3,
        eos_token_id=2,
    )

    assert request.status == RequestStatus.WAITING
    request.start()
    request.append_token(30)

    assert request.token_ids == [10, 20, 30]
    assert request.num_generated_tokens == 1
    assert not request.is_finished

    request.append_token(2)
    assert request.is_finished
    assert request.reason == "eos"
    assert request.generated_token_ids == [30, 2]
    
if __name__ == "__main__":
    main()
