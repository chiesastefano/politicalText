import re
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from collections import Counter
import subprocess
import sys
import torch
import shap
from transformers import pipeline


def process_debate_transcripts(input_dir, output_dir):
    """
    Process all debate transcript .txt files in the input directory and export structured
    dialogue data to CSV files in the output directory.

    Each .txt file is expected to follow the naming format:
    SPEAKER1_SPEAKER2_LOCATION_YEAR.txt

    Parameters:
    - input_dir (str or Path): Directory containing raw transcript .txt files.
    - output_dir (str or Path): Directory where processed CSV files will be saved.

    For each file:
    - Parses metadata from the filename (speakers, location, year)
    - Extracts dialogue lines (e.g., "CANDIDATE: ...")
    - Cleans and segments speech into ≤35-word chunks without breaking sentence structure
    - Classifies unknown speakers as "Moderator"
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for txt_file in input_path.glob("*.txt"):
        # Extract metadata from filename
        parts = txt_file.stem.split("_")
        if len(parts) < 4:
            print(f"Skipping {txt_file.name} — unexpected filename format.")
            continue

        speaker1, speaker2 = parts[0].upper(), parts[1].upper()
        location = parts[2].capitalize()
        year = parts[3]

        with open(txt_file, "r", encoding="utf-8") as file:
            text = file.read()

        # Match dialogue lines with the format "NAME: speech content"
        dialogues = re.findall(r'([A-Z]+):([\s\S]*?)(?=(?:[A-Z]+:)|\Z)', text)

        data = []
        speech_id = 1

        for speaker, line in dialogues:
            # Clean up line: remove annotations, newlines, and extra spaces
            line = re.sub(r'\[[^\]]+\]', '', line)
            line = line.replace('\n', ' ').replace('\r', ' ').strip()
            line = ' '.join(line.split())

            if not line:
                continue

            # Default to "Moderator" if speaker not one of the expected two
            if speaker not in [speaker1, speaker2]:
                speaker = "Moderator"

            # Split speech into <=30 word segments while respecting sentence boundaries
            segments = split_speech_into_segments(line)

            for segment in segments:
                if segment.strip():
                    data.append({
                        "SpeechID": speech_id,
                        "Speech": segment.strip(),
                        "Speaker": speaker.title(),
                        "Location": location,
                        "Year": year
                    })
                    speech_id += 1

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        output_file = output_path / f"{txt_file.stem}.csv"
        df.to_csv(output_file, index=False)

        print(f"Processed: {txt_file.name} → {output_file.name}")


def split_speech_into_segments(speech):
    """
    Splits a block of speech into segments of up to 35 words each, preserving
    sentence boundaries where possible.

    If a sentence exceeds 35 words, it is chunked accordingly.

    Parameters:
    - speech (str): Full speech text.

    Returns:
    - List[str]: Segments of the speech, each ≤35 words.
    """
    # Split by sentence-ending punctuation followed by space and capital letter (or end of string)
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    sentences = re.split(sentence_pattern, speech)

    segments = []
    current_segment = ""

    for sentence in sentences:
        sentence_words = sentence.split()
        word_count = len(sentence_words)

        # Split long sentences directly into 35-word chunks
        if word_count > 35:
            if current_segment:
                segments.append(current_segment)
                current_segment = ""

            words = sentence.split()
            chunk = []

            for word in words:
                chunk.append(word)
                if len(chunk) == 35:
                    segments.append(" ".join(chunk))
                    chunk = []

            if chunk:
                current_segment = " ".join(chunk)
        else:
            # Check if adding the sentence would exceed the 35-word segment limit
            potential_segment = current_segment + (" " if current_segment else "") + sentence
            if len(potential_segment.split()) > 35:
                segments.append(current_segment)
                current_segment = sentence
            else:
                current_segment = potential_segment

    if current_segment:
        segments.append(current_segment)

    return segments


def merge_lowercase_speeches(speeches: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rows where 'Speech' starts with a lowercase letter or any dash
    (-, –, —) into the previous row, preserving other columns from the most
    recent valid row.

    Parameters:
    - speeches: pd.DataFrame with at least a column named 'Speech'.

    Returns:
    - pd.DataFrame with merged rows and all original columns.
    """
    # Match lowercase letters OR -, –, —
    starts_with_merge = speeches['Speech'].str.match(r'^[a-z\-–—]')
    merged_rows = []
    buffer_speech = ""
    current_row = None

    for is_merge, row in zip(starts_with_merge, speeches.itertuples(index=False)):
        if is_merge:
            buffer_speech += " " + row.Speech
        else:
            if buffer_speech and current_row is not None:
                current_row = current_row._replace(Speech=current_row.Speech + buffer_speech)
                merged_rows.append(current_row)
                buffer_speech = ""
            else:
                if current_row is not None:
                    merged_rows.append(current_row)
            current_row = row

    if buffer_speech and current_row is not None:
        current_row = current_row._replace(Speech=current_row.Speech + buffer_speech)
        merged_rows.append(current_row)
    elif current_row is not None:
        merged_rows.append(current_row)

    return pd.DataFrame(merged_rows, columns=speeches.columns)


def get_top_emotion(text, classifier):
    """
    Classify the top emotion in a text snippet using a HuggingFace pipeline classifier.

    Parameters:
    - text (str): Input text to classify.
    - classifier (pipeline): A HuggingFace emotion classifier pipeline.

    Returns:
    - pd.Series: A pandas Series with [top_emotion_label, top_score].
    """
    result = classifier(text)[0]  # No need to truncate manually
    top = max(result, key=lambda x: x['score'])
    return pd.Series([top['label'], top['score']])


def classify_speech_emotion(df, emotion=None, classifier=None):
    # If an emotion is provided, filter by it; if not, use the entire dataset
    if emotion:
        filtered_speeches = df[df['grouped_emotion'] == emotion]
    else:
        filtered_speeches = df

    # Create lists to store the results
    labels = []
    scores = []

    # Iterate over the rows in the dataframe
    for index, row in filtered_speeches.iterrows():
        sentence = row['Speech']

        # Classify the speech using the given classifier
        result = classifier(sentence)

        # Check if the result contains at least one entry
        if result:
            # Extract the label and score (confidence)
            label = result[0]['label']
            score = result[0]['score']

            # Append the results to the lists
            labels.append(label)
            scores.append(score)
        else:
            # If result is empty, append None values
            labels.append(None)
            scores.append(None)

    # Add the results as new columns to the dataframe
    df['classified_label'] = labels
    df['classification_score'] = scores

    return df


def compute_relative_emotion_frequency(
        df,
        speaker=None,
        emotion=None,
        year=None,
        location=None,
        emotion_col="emotion",
        democrat=0
):
    """
    Compute the relative frequency of each emotion (or grouped_emotion) for each year, speaker, and location,
    then optionally filter the output for display.

    Parameters:
    - df (pd.DataFrame): DataFrame with at least 'Year', 'Speaker', 'Location', and emotion columns.
    - speaker (str, optional): If provided, filter output by this speaker.
    - emotion (str, optional): If provided, filter output by this emotion/grouped_emotion.
    - year (str or int, optional): If provided, filter output by this year.
    - location (str, optional): If provided, filter output by this location.
    - emotion_col (str, optional): Column to use for emotion ('emotion' or 'grouped_emotion'). Default is 'emotion'.
    - democrat (int, optional): If 1, only include rows where democrat==1.

    Returns:
    - pd.DataFrame: DataFrame with columns ['Year', 'Speaker', 'Location', emotion_col, 'relative_frequency'].
    """
    if democrat == 1:
        df = df[df['democrat'] == 1]

    group_cols = ['Year', 'Speaker', 'Location']
    freq = (
        df.groupby(group_cols)[emotion_col]
        .value_counts(normalize=True)
        .rename('relative_frequency')
        .reset_index()
    )

    # Now apply filters for display only
    if speaker is not None:
        freq = freq[freq['Speaker'] == speaker]
    if emotion is not None:
        freq = freq[freq[emotion_col] == emotion]
    if year is not None:
        freq = freq[freq['Year'] == year]
    if location is not None:
        freq = freq[freq['Location'] == location]

    return freq


def plot_classified_label_frequencies(speeches_done, speaker, year, emotion):
    """
    Plot the relative frequencies of the 'classified_label' for a given speaker, year, and emotion.

    Parameters:
    - speeches_done (pd.DataFrame): DataFrame containing the speech data with columns 'Speaker', 'grouped_emotion', 'Year', and 'classified_label'.
    - speaker (str): The speaker whose data to plot.
    - year (int or str): The year to filter the data by.
    - emotion (str): The emotion to filter the data by.

    Returns:
    - None: Displays a bar plot of the relative frequencies of classified labels.
    """

    # Filter the DataFrame based on the provided speaker, year, and emotion
    filtered_speeches = speeches_done[
        (speeches_done['Speaker'] == speaker) &
        (speeches_done['Year'] == year) &
        (speeches_done['grouped_emotion'] == emotion)
        ]

    # If no data is found after filtering, print a message and return
    if filtered_speeches.empty:
        print(f"No data found for {speaker} in {year} with emotion {emotion}.")
        return

    # Compute the relative frequencies of 'classified_label'
    classified_label_freq = filtered_speeches['classified_label'].value_counts(normalize=True).reset_index()
    classified_label_freq.columns = ['classified_label', 'relative_frequency']

    # Plot the relative frequencies (updated code to avoid deprecation warning)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='relative_frequency', y='classified_label', data=classified_label_freq, hue='classified_label',
                palette='viridis', legend=False)
    plt.title(f'Relative Frequencies of Classified Labels for {speaker} ({year}) - Emotion: {emotion}')
    plt.xlabel('Relative Frequency')
    plt.ylabel('Classified Label')
    plt.show()




def plot_most_frequent_words(speeches_done, speaker, year, emotion=None, top_n=20):
    """
    Plot the most frequent non-stop words using spaCy for a given speaker, year, and optionally emotion.
    """
    # Ensure the spaCy model is available
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Model 'en_core_web_sm' not found. Downloading now...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("Model downloaded successfully.")
        nlp = spacy.load("en_core_web_sm")

    # Filter speeches by speaker and year
    filtered_speeches = speeches_done[
        (speeches_done['Speaker'] == speaker) &
        (speeches_done['Year'] == year)
    ]

    # Optionally filter by emotion
    if emotion is not None:
        filtered_speeches = filtered_speeches[filtered_speeches['grouped_emotion'] == emotion]

    if filtered_speeches.empty:
        print(f"No data found for {speaker} in {year}" + (f" with emotion {emotion}." if emotion else "."))
        return

    # Combine and process text
    text = ' '.join(filtered_speeches["Speech"]).lower()
    doc = nlp(text)

    # Filter tokens
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop # to remove stop words
        and not token.is_punct # to remove punctuation
        and not token.is_space # to remove spaces
        and not token.like_num # to remove numbers
        and len(token.text) > 2 # to remove short words
    ]

    word_freq = Counter(tokens)
    most_common = word_freq.most_common(top_n)
    freq_df = pd.DataFrame(most_common, columns=['word', 'frequency'])

    plt.figure(figsize=(10, 6))
    sns.barplot(y='word', x='frequency', data=freq_df, hue='word', palette='viridis', legend=False)
    title_emotion = f" - Emotion: {emotion}" if emotion else ""
    plt.title(f'Most Frequent Words for {speaker} ({year}){title_emotion}')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.show()


def get_word_emotion_distribution(speeches_df, word, speaker=None, year=None, emotion=None):
    """
    Given a word, return the frequency distribution of grouped_emotion labels
    in which the word appears in the speeches, optionally filtered by speaker, year, or emotion.
    If emotion is specified, prints the sentences containing the word with that emotion.

    Parameters:
    - speeches_df (pd.DataFrame): DataFrame with 'Speech', 'grouped_emotion', 'Speaker', and 'Year' columns.
    - word (str): The target word to analyze.
    - speaker (str or None): Filter by speaker if specified.
    - year (int or str or None): Filter by year if specified.
    - emotion (str or None): If set, prints the sentences containing the word with that emotion.

    Returns:
    - pd.DataFrame: Emotion distribution for the word.
    """
    # Ensure spaCy model is installed
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    # Preprocess target word
    target_doc = nlp(word.lower())
    if not target_doc:
        print("Invalid input word.")
        return pd.DataFrame()
    target_lemma = target_doc[0].lemma_

    # Filter speeches
    df = speeches_df.dropna(subset=["Speech", "grouped_emotion"])
    if speaker:
        df = df[df["Speaker"] == speaker]
    if year:
        df = df[df["Year"] == year]

    # Count emotions where the word appears
    emotion_counter = Counter()
    matching_sentences = []

    for _, row in df.iterrows():
        doc = nlp(row["Speech"].lower())
        lemmas = {token.lemma_ for token in doc
                  if not token.is_stop and not token.is_punct and not token.like_num and len(token.text) > 2}

        if target_lemma in lemmas:
            label = row["grouped_emotion"]
            emotion_counter[label] += 1

            # If user asked for sentences with a specific emotion
            if emotion is not None and label == emotion:
                matching_sentences.append(row["Speech"].strip())

    # Print matching sentences if requested
    if emotion is not None and matching_sentences:
        print(f"\nSentences containing the word '{word}' in speeches labeled with emotion '{emotion}':\n")
        for sentence in matching_sentences:
            print(f"- {sentence}")

    # Return emotion distribution
    result_df = pd.DataFrame(emotion_counter.items(), columns=["Emotion", "Frequency"])
    result_df.sort_values("Frequency", ascending=False, inplace=True)
    return result_df


def analyze_emotions_with_attention(sentence, model, tokenizer, emotion_map=None):
    """
    Analyze emotions in a sentence using a transformer model.
    Displays top predicted emotions, attention heatmap, and SHAP token importance.

    Parameters
    ----------
    sentence : str
        Input sentence.
    model : transformers.PreTrainedModel
        Hugging Face model for emotion classification.
    tokenizer : transformers.PreTrainedTokenizer
        Corresponding tokenizer.
    emotion_map : dict, optional
        Optional mapping of raw emotions to broader categories.

    Returns
    -------
    None
    """

    # Get emotion labels
    id2label = model.config.id2label

    # Tokenize input
    inputs = tokenizer(sentence, return_tensors="pt") #PyTorch tensors
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Forward pass to get predictions and attentions
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions
        logits = outputs.logits

    # Top-5 predicted emotions
    probs = torch.softmax(logits, dim=1)[0]
    topk = torch.topk(probs, k=5)

    print("Top predicted emotions:")
    for score, idx in zip(topk.values, topk.indices):
        raw_emotion = id2label[idx.item()]
        mapped_emotion = emotion_map.get(raw_emotion, raw_emotion) if emotion_map else raw_emotion
        print(f"{raw_emotion} -> {mapped_emotion}: {score.item():.3f}")

    # Attention heatmap
    last_layer_attention = attentions[-1][0]  # (num_heads, seq_len, seq_len)
    attention_weights = last_layer_attention[0].detach().cpu().numpy()  # first head
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labeled_tokens = [f"{i}:{tok}" for i, tok in enumerate(tokens)]

    plt.figure(figsize=(16, 12))  # Larger figure for readability
    sns.heatmap(attention_weights,
                xticklabels=labeled_tokens,
                yticklabels=labeled_tokens,
                cmap="viridis")
    plt.title("Attention heatmap (last layer, head 0)", fontsize=14)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

    # --- SHAP token attribution ---
    print("\nRunning SHAP analysis (this may take a few seconds)...")

    # Build a pipeline for shap
    clf_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

    # Create SHAP explainer
    explainer = shap.Explainer(clf_pipeline)

    # Run SHAP on the sentence
    shap_values = explainer([sentence])

    # Plot SHAP explanation
    shap.plots.text(shap_values[0])
