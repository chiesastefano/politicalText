import re
import pandas as pd
from pathlib import Path

def process_debate_transcripts(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for txt_file in input_path.glob("*.txt"):
        # Parse metadata from filename
        parts = txt_file.stem.split("_")
        if len(parts) < 4:
            print(f"Skipping {txt_file.name} — unexpected filename format.")
            continue

        speaker1, speaker2 = parts[0].upper(), parts[1].upper()
        location = parts[2].capitalize()
        year = parts[3]

        with open(txt_file, "r", encoding="utf-8") as file:
            text = file.read()

        # Extract dialogue lines with full-caps names
        dialogues = re.findall(r'([A-Z]+):([\s\S]*?)(?=(?:[A-Z]+:)|\Z)', text)

        data = []

        for idx, (speaker, line) in enumerate(dialogues, start=1):
            # Remove [bracketed annotations] and all newline characters
            line = re.sub(r'\[[^\]]+\]', '', line)  # Remove [bracketed]
            line = line.replace('\n', ' ').replace('\r', ' ').strip()  # Remove newlines
            line = ' '.join(line.split())  # Normalize spaces

            if speaker not in [speaker1, speaker2]:
                speaker = "Moderator"

            if line:
                data.append({
                    "SpeechID": idx,
                    "Speech": line,
                    "Speaker": speaker.title(),
                    "Location": location,
                    "Year": year
                })

        df = pd.DataFrame(data)

        output_file = output_path / f"{txt_file.stem}.csv"
        df.to_csv(output_file, index=False)

        print(f"Processed: {txt_file.name} → {output_file.name}")
