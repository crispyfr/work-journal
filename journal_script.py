import configparser
import datetime
import os
import tempfile
import numpy as np
import whisper
from scipy.io import wavfile
import sounddevice as sd
import json
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from collections import Counter
import nltk
import psutil
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import signal
import gc
import sqlite3

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Constants
LANGUAGES = ["English", "Albanian", "French", "German", "Italian", "Spanish"]
MODELS = ["tiny", "base", "small", "medium", "large"]


# Global settings
config = configparser.ConfigParser()
if not os.path.exists('work_journal_config.ini'):
    config['DEFAULT'] = {
        'language': 'English',
        'model': 'small',
        'duration': '60',
        'auto_tag': 'True'
    }
    with open('work_journal_config.ini', 'w') as configfile:
        config.write(configfile)
config.read('work_journal_config.ini')

def get_settings():
    return {
        "language": config['DEFAULT'].get('language', 'English'),
        "model": config['DEFAULT'].get('model', 'small'),
        "duration": int(config['DEFAULT'].get('duration', '60')),
        "auto_tag": config['DEFAULT'].getboolean('auto_tag', True)
    }

settings = {
    "language": "English",
    "model": "small",
    "duration": 60,
    "auto_tag": True
}

# Global variable to track if recording should stop
stop_recording = False

class JournalDatabase:
    def __init__(self, db_file='work_journal.db'):
        self.conn = sqlite3.connect(db_file)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                text TEXT,
                tags TEXT
            )
        ''')
        self.conn.commit()

    def save_entry(self, text, tags, timestamp=None):
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO entries (timestamp, text, tags)
            VALUES (?, ?, ?)
        ''', (timestamp, text, json.dumps(tags)))
        self.conn.commit()
        return cursor.lastrowid

    def get_entries(self, limit=None, offset=0):
        cursor = self.conn.cursor()
        if limit:
            cursor.execute('''
                SELECT * FROM entries
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))
        else:
            cursor.execute('SELECT * FROM entries ORDER BY timestamp DESC')
        return [
            {
                'id': row[0],
                'timestamp': row[1],
                'text': row[2],
                'tags': json.loads(row[3])
            }
            for row in cursor.fetchall()
        ]

    def search_entries(self, search_term, search_type='text'):
        cursor = self.conn.cursor()
        if search_type == 'text':
            cursor.execute('''
                SELECT * FROM entries
                WHERE text LIKE ?
                ORDER BY timestamp DESC
            ''', (f'%{search_term}%',))
        else:  # search by tag
            cursor.execute('''
                SELECT * FROM entries
                WHERE tags LIKE ?
                ORDER BY timestamp DESC
            ''', (f'%{search_term}%',))
        return [
            {
                'id': row[0],
                'timestamp': row[1],
                'text': row[2],
                'tags': json.loads(row[3])
            }
            for row in cursor.fetchall()
        ]

    def update_entry(self, entry_id, text, tags):
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE entries
            SET text = ?, tags = ?
            WHERE id = ?
        ''', (text, json.dumps(tags), entry_id))
        self.conn.commit()

    def remove_entry(self, entry_id):
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM entries WHERE id = ?', (entry_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()

# Global instance of JournalDatabase
db = JournalDatabase()

def signal_handler(signum, frame):
    global stop_recording
    stop_recording = True
    print("\nRecording stopped.")

class TranscriptionTool:
    def __init__(self, model_name):
        print(f"Loading {model_name} model... This may take a moment.")
        self.model = whisper.load_model(model_name)
        print("Model loaded and ready for use.")

    def transcribe_audio(self, audio_file, language):
        print("Transcribing...")
        return self.model.transcribe(audio_file, language=language, fp16=False)

    def __del__(self):
        del self.model
        gc.collect()
        print("Model unloaded from memory.")

# Global instance of TranscriptionTool
transcription_tool = None

def record_audio(duration, samplerate=16000):
    global stop_recording
    stop_recording = False
    
    print(f"Recording for up to {duration} seconds... Press Ctrl+C to stop recording.")
    
    signal.signal(signal.SIGINT, signal_handler)
    
    recorded_audio = []
    try:
        with sd.InputStream(samplerate=samplerate, channels=1) as stream:
            for _ in range(int(samplerate * duration)):
                if stop_recording:
                    break
                audio_chunk, overflowed = stream.read(1)
                if not overflowed:
                    recorded_audio.append(audio_chunk[0])
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    return np.array(recorded_audio)

def is_silent(audio_array, threshold=0.01):
    return np.max(np.abs(audio_array)) < threshold

def extract_keywords(text, num_keywords=5):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_tokens = [w for w in word_tokens if w not in stop_words and w.isalnum()]
    return [word for word, _ in Counter(filtered_tokens).most_common(num_keywords)]

def get_resource_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    disk = psutil.disk_usage('/')
    disk_percent = disk.percent
    return cpu_percent, memory_percent, disk_percent

def sanitize_text(text):
    return unidecode(text)

def summarize_day(date):
    entries = [entry for entry in db.get_entries() if entry['timestamp'].startswith(date.isoformat())]
    
    if not entries:
        return None

    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, sanitize_text(f"Work Journal Summary for {date}"), align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)

    for entry in entries:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, sanitize_text(f"Entry at {entry['timestamp']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 5, sanitize_text(f"Tags: {', '.join(entry['tags'])}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, sanitize_text(entry['text']))
        pdf.ln(10)
    
    summary_filename = f"summary_{date}.pdf"
    pdf.output(summary_filename)
    return summary_filename

def main_menu():
    global transcription_tool
    
    transcription_tool = TranscriptionTool(settings['model'])

    while True:
        print("\nWork Journal")
        print("1. Record New Audio Entry")
        print("2. Add New Text Entry")
        print("3. List Entries")
        print("4. Search Entries")
        print("5. Summarize Your Day")
        print("6. Display Resource Usage")
        print("7. Change Settings")
        print("8. Exit")
        
        choice = input("Enter your choice (1-8): ")
        
        if choice == '1':
            record_new_entry()
        elif choice == '2':
            add_text_entry()
        elif choice == '3':
            list_entries()
        elif choice == '4':
            search_entries()
        elif choice == '5':
            summarize_day_menu()
        elif choice == '6':
            display_resource_usage()
        elif choice == '7':
            change_settings()
        elif choice == '8':
            print("Exiting Work Journal. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

    del transcription_tool
    gc.collect()
    db.close()

def record_new_entry():
    print("\nRecord New Audio Entry")
    manual_tags = input("Additional tags (comma-separated): ")

    print(f"Recording for up to {settings['duration']} seconds... Press Ctrl+C to stop recording.")
    audio = record_audio(settings['duration'])
    
    if is_silent(audio):
        print("The audio appears to be silent. You may want to check your microphone and try again.")
    else:
        print("Transcribing...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_filename = temp_audio.name
            wavfile.write(temp_filename, 16000, audio)

        try:
            result = transcription_tool.transcribe_audio(temp_filename, settings['language'])
            print("Transcription complete!")
            transcribed_text = result["text"]
            print("Transcription:", transcribed_text)
            print("Language:", result["language"])
            
            tags = []
            if settings['auto_tag']:
                tags = extract_keywords(transcribed_text)
            if manual_tags:
                tags.extend([tag.strip() for tag in manual_tags.split(',')])
            tags = list(set(tags))  # Remove duplicates
            
            db.save_entry(transcribed_text, tags)
            print(f"Entry saved successfully with tags: {', '.join(tags)}")
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
        finally:
            os.unlink(temp_filename)

def add_text_entry():
    print("\nAdd New Text Entry")
    text = input("Enter your text entry: ")
    manual_tags = input("Enter tags (comma-separated): ")

    tags = []
    if settings['auto_tag']:
        tags = extract_keywords(text)
    if manual_tags:
        tags.extend([tag.strip() for tag in manual_tags.split(',')])
    tags = list(set(tags))  # Remove duplicates

    db.save_entry(text, tags)
    print(f"Entry saved successfully with tags: {', '.join(tags)}")

def list_entries():
    entries = db.get_entries()
    if not entries:
        print("No entries found.")
        return

    page_size = 5
    current_page = 0

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n=== Work Journal Entries ===")
        
        start = current_page * page_size
        end = start + page_size
        current_entries = entries[start:end]

        for i, entry in enumerate(current_entries, start=start+1):
            print(f"\n[{i}] Entry from {entry.get('timestamp', 'Unknown date')}")
            print(f"Tags: {', '.join(entry.get('tags', []))}")
            print(f"Text: {entry.get('text', '')[:100]}{'...' if len(entry.get('text', '')) > 100 else ''}")

        print("\n" + "="*30)
        print(f"Page {current_page + 1} of {(len(entries) - 1) // page_size + 1}")
        print("\nOptions:")
        print("  [n] Next page")
        print("  [p] Previous page")
        print("  [v] View full entry")
        print("  [e] Edit an entry")
        print("  [r] Remove an entry")
        print("  [q] Return to main menu")

        choice = input("\nEnter your choice: ").lower()

        if choice == 'n' and end < len(entries):
            current_page += 1
        elif choice == 'p' and current_page > 0:
            current_page -= 1
        elif choice == 'v':
            entry_num = int(input("Enter the number of the entry to view: ")) - 1
            if 0 <= entry_num < len(entries):
                os.system('cls' if os.name == 'nt' else 'clear')
                entry = entries[entry_num]
                print(f"\nFull Entry from {entry.get('timestamp', 'Unknown date')}")
                print(f"Tags: {', '.join(entry.get('tags', []))}")
                print(f"Text: {entry.get('text', '')}")
                input("\nPress Enter to continue...")
            else:
                print("Invalid entry number.")
        elif choice == 'e':
            entry_num = int(input("Enter the number of the entry to edit: ")) - 1
            if 0 <= entry_num < len(entries):
                entry = entries[entry_num]
                new_text = input("Enter new text (press Enter to keep current): ") or entry.get('text', '')
                new_tags = input("Enter new tags (comma-separated, press Enter to keep current): ") or ', '.join(entry.get('tags', []))
                db.update_entry(entry['id'], new_text, new_tags.split(', '))
                print("Changes saved successfully.")
                entries = db.get_entries()  # Refresh the entries list
            else:
                print("Invalid entry number.")
        elif choice == 'r':
            entry_num = int(input("Enter the number of the entry to remove: ")) - 1
            if 0 <= entry_num < len(entries):
                entry = entries[entry_num]
                db.remove_entry(entry['id'])
                print("Entry removed successfully.")
                entries = db.get_entries()  # Refresh the entries list
            else:
                print("Invalid entry number.")
        elif choice == 'q':
            print("Returning to main menu.")
            break
        else:
            print("Invalid choice. Please try again.")

        input("\nPress Enter to continue...")

def search_entries():
    print("\nSearch Entries")
    search_type = input("Search by (1. Text, 2. Tag): ")
    search_term = input("Enter search term: ")
    
    if search_type == "1":
        results = db.search_entries(search_term, search_type='text')
    else:
        results = db.search_entries(search_term, search_type='tag')
    
    print(f"Found {len(results)} entries:")
    for entry in results:
        print(f"\nEntry from {entry.get('timestamp', 'Unknown date')}")
        print(f"Tags: {', '.join(entry.get('tags', []))}")
        print(f"Text: {entry.get('text', '')}")

def summarize_day_menu():
    print("\nSummarize Your Day")
    date_str = input("Enter date to summarize (YYYY-MM-DD, default today): ") or datetime.date.today().isoformat()
    try:
        date = datetime.date.fromisoformat(date_str)
        summary_file = summarize_day(date)
        if summary_file:
            print(f"Summary generated for {date}")
            print(f"Summary saved as: {summary_file}")
        else:
            print(f"No entries found for {date}")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")

def display_resource_usage():
    cpu_percent, memory_percent, disk_percent = get_resource_usage()
    print("\nResource Usage")
    print(f"CPU Usage: {cpu_percent:.1f}%")
    print(f"Memory Usage: {memory_percent:.1f}%")
    print(f"Disk Usage: {disk_percent:.1f}%")

def change_settings():
    global settings, transcription_tool
    print("Current Settings:")
    for key, value in settings.items():
        print(f"{key.capitalize()}: {value}")

    print("Change Settings:")
    
    # Language
    print("Select language:")
    for i, lang in enumerate(LANGUAGES, 1):
        print(f"{i}. {lang}")
    lang_choice = input(f"Enter your choice (1-{len(LANGUAGES)}, or press Enter to keep current): ")
    if lang_choice:
        config['DEFAULT']['language'] = LANGUAGES[int(lang_choice) - 1]

    # Model
    print("Select model:")
    for i, model in enumerate(MODELS, 1):
        print(f"{i}. {model}")
    model_choice = input(f"Enter your choice (1-{len(MODELS)}, or press Enter to keep current): ")
    if model_choice:
        new_model = MODELS[int(model_choice) - 1]
        if new_model != settings['model']:
            config['DEFAULT']['model'] = new_model
            del transcription_tool
            transcription_tool = TranscriptionTool(new_model)

    # Duration
    duration = input("Enter recording duration in seconds (or press Enter to keep current): ")
    if duration:
        config['DEFAULT']['duration'] = duration

    # Auto-tag
    auto_tag = input("Enable auto-tagging? (y/n, or press Enter to keep current): ").lower()
    if auto_tag in ['y', 'n']:
        config['DEFAULT']['auto_tag'] = str(auto_tag == 'y')

    # Save the updated config
    with open('work_journal_config.ini', 'w') as configfile:
        config.write(configfile)

    # Update the settings
    settings.update(get_settings())

    print("Updated Settings:")
    for key, value in settings.items():
        print(f"{key.capitalize()}: {value}")
        
if __name__ == "__main__":
    main_menu()