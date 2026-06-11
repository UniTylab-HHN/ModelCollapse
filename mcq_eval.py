import torch
import pandas as pd
import gc
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re

def format_prompt(row, prompt_style="strict"):
    """Format a single MCQ question as a prompt.

    prompt_style options:
    - "strict": Forces model to output ONLY a number 
    - "concise": Short prompt asking for the number
    - "cot": Chain-of-thought style
    """
    case = str(row.get('Case', '')) if pd.notna(row.get('Case', '')) else ''
    question = str(row.get('Question', '')) if pd.notna(row.get('Question', '')) else ''

    # Get answer choices
    choices = []
    for i in range(1, 6):
        col = f'Answer {i}'
        if col in row and pd.notna(row[col]):
            choice_text = str(row[col])
            # Remove any existing numbering like "0) " at the start
            if choice_text and len(choice_text) > 2 and choice_text[1] == ')':
                choice_text = choice_text[3:].strip()
            choices.append(choice_text)

    # Build choices string
    choices_str = ""
    for i, choice in enumerate(choices):
        choices_str += f"{i}. {choice}\n"

    if prompt_style == "strict":
        # STRICT: Force model to output ONLY a single digit with few-shot examples
        prompt = f"""Answer medical multiple choice questions with ONLY the option number (0-4).

                    Example 1:
                    What is the capital of France?
                    0. Berlin
                    1. Paris
                    2. London
                    Answer: 1

                    Example 2:
                    What color is blood?
                    0. Blue
                    1. Green
                    2. Red
                    Answer: 2

                    Now answer this question:
                    {case}
                    {question}

                    {choices_str}
                    Answer:"""

    elif prompt_style == "concise":
        # Concise prompt
        prompt = f"""Question: {case} {question}

{choices_str}
Reply with only the number (0-4):"""

    elif prompt_style == "cot":
        # Chain-of-thought - let model explain then extract number
        prompt = f"""{case}

{question}

{choices_str}
Think step by step and give your final answer as a number (0-4):"""

    else:
        # Verbose German style
        prompt = f"""Beantworte mit NUR einer Zahl (0, 1, 2, 3 oder 4). Keine Erklärung!

                        {case}

                        {question}

                        {choices_str}
                    Antwort:"""

    return prompt, choices

PROMPT_STYLE = "strict"  # Options: "strict", "concise", "cot", "verbose"

def get_model_answer(model, tokenizer, prompt, max_new_tokens=10, use_chat_template=False):
    """Get model's answer for a prompt. Few-shot examples should give cleaner outputs."""

    # For Instruct models, use chat template
    if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
    else:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only the new tokens
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Extract the answer (0-4)
    answer = extract_answer(generated_text)
    return answer, generated_text.strip()

def extract_answer(text):
    """Extract answer index (0-4) from model output.

    Handles various formats:
    - "2" (just the number)
    - "2." or "(2)"
    - "Die Antwort ist 2"
    - "Answer: 2"
    - "Option 2"
    """
    
    text = text.strip()

    # If empty, return -1
    if not text:
        return -1

    # Strategy 1: If first character is a digit 0-4, use it
    if text[0] in '01234':
        return int(text[0])

    # Strategy 2: Look for common patterns
    patterns = [
        r'(?:answer|antwort|option|nummer|number)[:\s]*([0-4])',  # "Answer: 2", "Antwort: 3"
        r'(?:ist|is)[:\s]*([0-4])',  # "ist 2", "is 3"
        r'\(([0-4])\)',  # "(2)"
        r'^([0-4])[.\s\)]',  # "2." or "2 " at start
        r'(?:^|\s)([0-4])(?:\s|$|\.)',  # standalone digit
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))

    # Strategy 3: Find ANY digit 0-4 in the text (last resort)
    for char in text:
        if char in '01234':
            return int(char)

    # If no valid answer found
    return -1

def evaluate_model(model, tokenizer, df, model_name, debug_print=False, use_chat_template=False):
    """Evaluate model on all questions and save detailed log.

    Args:
        use_chat_template: Set True for Instruct models (e.g., Llama3-Instruct)
    """
    results = []
    log_entries = []  # For the log file
    correct = 0
    total = len(df)

    # Auto-enable debug printing if DEBUG_MODE is on
    if 'DEBUG_MODE' in globals() and DEBUG_MODE:
        debug_print = True

    print(f"\n Evaluating {model_name} on {total} questions...")
    if use_chat_template:
        print("Using chat template for Instruct model")
    if debug_print:
        print("DEBUG MODE: Will print details for each question\n")

    for idx, row in tqdm(df.iterrows(), total=total, desc="Evaluating", disable=debug_print):
        prompt, choices = format_prompt(row, prompt_style=PROMPT_STYLE)

        # Get correct answer
        correct_idx = int(row['Correct Answer']) if pd.notna(row['Correct Answer']) else -1
        correct_text = choices[correct_idx] if 0 <= correct_idx < len(choices) else "N/A"

        # Get model's answer (use chat template for Instruct models)
        model_idx, raw_output = get_model_answer(model, tokenizer, prompt, use_chat_template=use_chat_template)
        model_text = choices[model_idx] if 0 <= model_idx < len(choices) else f"Invalid ({raw_output})"

        # Check if correct
        is_correct = (model_idx == correct_idx)
        if is_correct:
            correct += 1

        # DEBUG PRINT - Show exactly what we sent and received
        if debug_print:
            print("=" * 70)
            print(f"QUESTION {idx + 1}/{total}")
            print("=" * 70)
            print("\nPROMPT SENT TO MODEL:")
            print("-" * 40)
            print(prompt)
            print("-" * 40)
            print(f"\nMODEL RAW OUTPUT: \"{raw_output}\"")
            print(f"Extracted Answer: {model_idx}")
            print(f"Correct Answer: {correct_idx}")
            print(f"{'CORRECT!' if is_correct else 'WRONG!'}")
            print()

        # Save to results
        results.append({
            'Question_No': idx + 1,
            'Case': str(row.get('Case', ''))[:100] + '...' if len(str(row.get('Case', ''))) > 100 else str(row.get('Case', '')),
            'Question': str(row.get('Question', '')),
            'Choices': ' | '.join([f"{i}) {c}" for i, c in enumerate(choices)]),
            'Correct_Answer_Index': correct_idx,
            'Correct_Answer': correct_text,
            'Model_Prediction_Index': model_idx,
            'Model_Answer': model_text,
            'Model_Raw_Output': raw_output,
            'Is_Correct': is_correct,
            'Model': model_name
        })

        # Save to log (full prompt + response) - DETAILED
        log_entries.append({
            'Question_No': idx + 1,
            'Input_Prompt': prompt,  # Exactly what we sent
            'Output_Raw': raw_output,  # Exactly what model returned
            'Output_Extracted_Index': model_idx,  # What we extracted as answer
            'Correct_Index': correct_idx,
            'Is_Correct': is_correct,
            'Model': model_name
        })

        # Print progress every 50 questions (only if not debug mode)
        if not debug_print and (idx + 1) % 50 == 0:
            print(f"   Progress: {idx+1}/{total} | Accuracy so far: {correct/(idx+1)*100:.1f}%")

    accuracy = correct / total * 100
    print(f"\n{'=' * 60}")
    print(f"{model_name} Evaluation Complete!")
    print(f"   Total: {total} | Correct: {correct} | Accuracy: {accuracy:.2f}%")
    print(f"{'=' * 60}")

    # Save log file
    log_df = pd.DataFrame(log_entries)
    log_file = f"{model_name}_prompts_log.csv"
    log_df.to_csv(log_file, index=False, encoding='utf-8-sig')
    print(f"Log saved to: {log_file}")

    return pd.DataFrame(results), accuracy, log_file