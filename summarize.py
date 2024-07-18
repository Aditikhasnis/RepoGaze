from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def summarize_text(text, model_name, token, max_length=512):
    # Load the summarization model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    # Summarization pipeline with GPU support if available
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

    # Split the text into chunks of max_length tokens
    inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding=True, stride=100, return_overflowing_tokens=True)
    summaries = []
    for input_ids in inputs['input_ids']:
        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        summary = summarizer(input_text, max_length=200, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    # Join the summaries
    final_summary = ' '.join(summaries)
    
    return final_summary
