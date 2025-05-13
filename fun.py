import re
import pandas as pd
from pathlib import Path


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
    - Cleans and segments speech into ≤30-word chunks without breaking sentence structure
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
    - List[str]: Segments of the speech, each ≤30 words.
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
                if len(chunk) == 30:
                    segments.append(" ".join(chunk))
                    chunk = []

            if chunk:
                current_segment = " ".join(chunk)
        else:
            # Check if adding the sentence would exceed the 30-word segment limit
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