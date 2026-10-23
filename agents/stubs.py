from agents.contracts import Candidate, SentenceResult, VerifyResult, Symbol, PredictionRequest
from agents.gemma_client import GemmaClient

# Initialize client in colab mode
client = GemmaClient(mode="colab")

def stub_prediction_agent(request: PredictionRequest) -> list[Candidate]:
    return [Candidate(symbol=Symbol(symbol_id="1", text="want"), score=0.9)]

def stub_sentence_agent(sequence: list[Symbol]) -> SentenceResult:
    prompt = "Expand these symbols into a sentence: " + ", ".join([s.text for s in sequence])
    output_text = client.generate(prompt)
    return SentenceResult(text=output_text or "I want water.", attempt=1)

def stub_verifier_agent(sentence: str, sequence: list[Symbol]) -> VerifyResult:
    return VerifyResult(ok=True)

if __name__ == "__main__":
    print("Testing end-to-end run on stubs...")
    req = PredictionRequest(sequence=[Symbol(symbol_id="1", text="want")], prev_turn=[], user_id="test_user")
    
    preds = stub_prediction_agent(req)
    sent = stub_sentence_agent(req.sequence)
    ver = stub_verifier_agent(sent.text, req.sequence)
    
    print("Prediction Result:", preds)
    print("Sentence Result:", sent)
    print("Verifier Result:", ver)
