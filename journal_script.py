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
from colorama import init, Fore, Style
from tqdm import tqdm
import time

# Initialize colorama
init()

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

settings = get_settings()

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

db = JournalDatabase()

def signal_handler(signum, frame):
    global stop_recording
    stop_recording = True
    print(f"\n{Fore.YELLOW}Recording stopped.{Style.RESET_ALL}")

class TranscriptionTool:
    def __init__(self, model_name):
        print(f"{Fore.CYAN}Loading {model_name} model... This may take a moment.{Style.RESET_ALL}")
        self.model = whisper.load_model(model_name)
        print(f"{Fore.GREEN}Model loaded and ready for use.{Style.RESET_ALL}")

    def transcribe_audio(self, audio_file, language):
        print(f"{Fore.CYAN}Transcribing...{Style.RESET_ALL}")
        return self.model.transcribe(audio_file, language=language, fp16=False)

    def __del__(self):
        del self.model
        gc.collect()
        print(f"{Fore.YELLOW}Model unloaded from memory.{Style.RESET_ALL}")

# Global instance of TranscriptionTool
transcription_tool = None

def record_audio(duration, samplerate=16000):
    global stop_recording
    stop_recording = False
    
    print(f"{Fore.CYAN}Recording for up to {duration} seconds... Press Ctrl+C to stop recording.{Style.RESET_ALL}")
    
    signal.signal(signal.SIGINT, signal_handler)
    
    recorded_audio = []
    try:
        with sd.InputStream(samplerate=samplerate, channels=1) as stream:
            for _ in tqdm(range(int(samplerate * duration)), desc="Recording"):
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

def print_banner():
    print(f"{Fore.CYAN}")
    print(r"""
    __        __         _      _                           _ 
    \ \      / /__  _ __| | __ | | ___  _   _ _ __ _ __   _| |
     \ \ /\ / / _ \| '__| |/ / | |/ _ \| | | | '__| '_ \ / _` |
      \ V  V / (_) | |  |   <  | | (_) | |_| | |  | | | | (_| |
       \_/\_/ \___/|_|  |_|\_\ |_|\___/ \__,_|_|  |_| |_|\__,_|
    """)
    print(f"{Style.RESET_ALL}")

def confirm_action(prompt):
    while True:
        choice = input(f"{Fore.YELLOW}{prompt} (y/n): {Style.RESET_ALL}").lower()
        if choice in ['y', 'n']:
            return choice == 'y'
        print(f"{Fore.RED}Please enter 'y' or 'n'.{Style.RESET_ALL}")

def main_menu():
    global transcription_tool
    
    transcription_tool = TranscriptionTool(settings['model'])

    while True:
        print_banner()
        print(f"{Fore.GREEN}\nWork Journal{Style.RESET_ALL}")
        print("1. Record New Audio Entry")
        print("2. Add New Text Entry")
        print("3. List Entries")
        print("4. Search Entries")
        print("5. Summarize Your Day")
        print("6. Display Resource Usage")
        print("7. Change Settings")
        print("8. Exit")
        
        choice = input(f"{Fore.CYAN}Enter your choice (1-8): {Style.RESET_ALL}")
        
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
            if confirm_action("Are you sure you want to exit?"):
                print(f"{Fore.GREEN}Exiting Work Journal. Goodbye!{Style.RESET_ALL}")
                break
        else:
            print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")

    del transcription_tool
    gc.collect()
    db.close()

def record_new_entry():
    print(f"\n{Fore.GREEN}Record New Audio Entry{Style.RESET_ALL}")
    manual_tags = input("Additional tags (comma-separated): ")

    print(f"Recording for up to {settings['duration']} seconds... Press Ctrl+C to stop recording.")
    audio = record_audio(settings['duration'])
    
    if is_silent(audio):
        print(f"{Fore.YELLOW}The audio appears to be silent. You may want to check your microphone and try again.{Style.RESET_ALL}")
    else:
        print(f"{Fore.CYAN}Transcribing...{Style.RESET_ALL}")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_filename = temp_audio.name
            wavfile.write(temp_filename, 16000, audio)

        try:
            result = transcription_tool.transcribe_audio(temp_filename, settings['language'])
            print(f"{Fore.GREEN}Transcription complete!{Style.RESET_ALL}")
            transcribed_text = result["text"]
            print(f"Transcription: {transcribed_text}")
            print(f"Language: {result['language']}")
            
            tags = []
            if settings['auto_tag']:
                tags = extract_keywords(transcribed_text)
            if manual_tags:
                tags.extend([tag.strip() for tag in manual_tags.split(',')])
            tags = list(set(tags))  # Remove duplicates
            
            db.save_entry(transcribed_text, tags)
            print(f"{Fore.GREEN}Entry saved successfully with tags: {', '.join(tags)}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error during transcription: {str(e)}{Style.RESET_ALL}")
        finally:
            os.unlink(temp_filename)

def add_text_entry():
    print(f"\n{Fore.GREEN}Add New Text Entry{Style.RESET_ALL}")
    print("Enter your text entry (press Enter twice on a new line to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            if lines and lines[-1] == "":
                break
        lines.append(line)
    text = "\n".join(lines).strip()
    
    if not text:
        print(f"{Fore.YELLOW}No text entered. Entry not saved.{Style.RESET_ALL}")
        return

    manual_tags = input("Enter tags (comma-separated): ")

    tags = []
    if settings['auto_tag']:
        tags = extract_keywords(text)
    if manual_tags:
        tags.extend([tag.strip() for tag in manual_tags.split(',')])
    tags = list(set(tags))  # Remove duplicates

    db.save_entry(text, tags)
    print(f"{Fore.GREEN}Entry saved successfully with tags: {', '.join(tags)}{Style.RESET_ALL}")


def list_entries():
    entries = db.get_entries()
    if not entries:
        print(f"{Fore.YELLOW}No entries found.{Style.RESET_ALL}")
        return

    page_size = 5
    current_page = 0

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"\n{Fore.GREEN}=== Work Journal Entries ==={Style.RESET_ALL}")
        
        start = current_page * page_size
        end = start + page_size
        current_entries = entries[start:end]

        for i, entry in enumerate(current_entries, start=start+1):
            print(f"\n{Fore.CYAN}[{i}] Entry from {entry.get('timestamp', 'Unknown date')}{Style.RESET_ALL}")
            print(f"Tags: {', '.join(entry.get('tags', []))}")
            print(f"Text: {entry.get('text', '')[:100]}{'...' if len(entry.get('text', '')) > 100 else ''}")

        print("\n" + "="*30)
        print(f"Page {current_page + 1} of {(len(entries) - 1) // page_size + 1}")
        print(f"\n{Fore.YELLOW}Options:{Style.RESET_ALL}")
        print("  [n] Next page")
        print("  [p] Previous page")
        print("  [v] View full entry")
        print("  [e] Edit an entry")
        print("  [r] Remove an entry")
        print("  [q] Return to main menu")

        choice = input(f"\n{Fore.CYAN}Enter your choice: {Style.RESET_ALL}").lower()

        if choice == 'n' and end < len(entries):
            current_page += 1
        elif choice == 'p' and current_page > 0:
            current_page -= 1
        elif choice == 'v':
            entry_num = int(input("Enter the number of the entry to view: ")) - 1
            if 0 <= entry_num < len(entries):
                os.system('cls' if os.name == 'nt' else 'clear')
                entry = entries[entry_num]
                print(f"\n{Fore.GREEN}Full Entry from {entry.get('timestamp', 'Unknown date')}{Style.RESET_ALL}")
                print(f"Tags: {', '.join(entry.get('tags', []))}")
                print(f"Text: {entry.get('text', '')}")
                input("\nPress Enter to continue...")
            else:
                print(f"{Fore.RED}Invalid entry number.{Style.RESET_ALL}")
        elif choice == 'e':
            entry_num = int(input("Enter the number of the entry to edit: ")) - 1
            if 0 <= entry_num < len(entries):
                entry = entries[entry_num]
                new_text = input("Enter new text (press Enter to keep current): ") or entry.get('text', '')
                new_tags = input("Enter new tags (comma-separated, press Enter to keep current): ") or ', '.join(entry.get('tags', []))
                if confirm_action("Are you sure you want to update this entry?"):
                    db.update_entry(entry['id'], new_text, new_tags.split(', '))
                    print(f"{Fore.GREEN}Changes saved successfully.{Style.RESET_ALL}")
                    entries = db.get_entries()  # Refresh the entries list
            else:
                print(f"{Fore.RED}Invalid entry number.{Style.RESET_ALL}")
        elif choice == 'r':
            entry_num = int(input("Enter the number of the entry to remove: ")) - 1
            if 0 <= entry_num < len(entries):
                entry = entries[entry_num]
                if confirm_action("Are you sure you want to remove this entry?"):
                    db.remove_entry(entry['id'])
                    print(f"{Fore.GREEN}Entry removed successfully.{Style.RESET_ALL}")
                    entries = db.get_entries()  # Refresh the entries list
            else:
                print(f"{Fore.RED}Invalid entry number.{Style.RESET_ALL}")
        elif choice == 'q':
            print(f"{Fore.YELLOW}Returning to main menu.{Style.RESET_ALL}")
            break
        else:
            print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")

        input(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")

def search_entries():
    print(f"\n{Fore.GREEN}Search Entries{Style.RESET_ALL}")
    search_type = input(f"{Fore.CYAN}Search by (1. Text, 2. Tag): {Style.RESET_ALL}")
    search_term = input(f"{Fore.CYAN}Enter search term: {Style.RESET_ALL}")
    
    if search_type == "1":
        results = db.search_entries(search_term, search_type='text')
    else:
        results = db.search_entries(search_term, search_type='tag')
    
    print(f"{Fore.GREEN}Found {len(results)} entries:{Style.RESET_ALL}")
    for entry in results:
        print(f"\n{Fore.CYAN}Entry from {entry.get('timestamp', 'Unknown date')}{Style.RESET_ALL}")
        print(f"Tags: {', '.join(entry.get('tags', []))}")
        print(f"Text: {entry.get('text', '')}")

def summarize_day_menu():
    print(f"\n{Fore.GREEN}Summarize Your Day{Style.RESET_ALL}")
    date_str = input(f"{Fore.CYAN}Enter date to summarize (YYYY-MM-DD, default today): {Style.RESET_ALL}") or datetime.date.today().isoformat()
    try:
        date = datetime.date.fromisoformat(date_str)
        summary_file = summarize_day(date)
        if summary_file:
            print(f"{Fore.GREEN}Summary generated for {date}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Summary saved as: {summary_file}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No entries found for {date}{Style.RESET_ALL}")
    except ValueError:
        print(f"{Fore.RED}Invalid date format. Please use YYYY-MM-DD.{Style.RESET_ALL}")

def display_resource_usage():
    cpu_percent, memory_percent, disk_percent = get_resource_usage()
    print(f"\n{Fore.GREEN}Resource Usage{Style.RESET_ALL}")
    print(f"{Fore.CYAN}CPU Usage: {cpu_percent:.1f}%{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Memory Usage: {memory_percent:.1f}%{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Disk Usage: {disk_percent:.1f}%{Style.RESET_ALL}")

def change_settings():
    global settings, transcription_tool
    print(f"{Fore.GREEN}Current Settings:{Style.RESET_ALL}")
    for key, value in settings.items():
        print(f"{Fore.CYAN}{key.capitalize()}: {value}{Style.RESET_ALL}")

    print(f"\n{Fore.GREEN}Change Settings:{Style.RESET_ALL}")
    
    # Language
    print(f"{Fore.YELLOW}Select language:{Style.RESET_ALL}")
    for i, lang in enumerate(LANGUAGES, 1):
        print(f"{i}. {lang}")
    lang_choice = input(f"{Fore.CYAN}Enter your choice (1-{len(LANGUAGES)}, or press Enter to keep current): {Style.RESET_ALL}")
    if lang_choice:
        config['DEFAULT']['language'] = LANGUAGES[int(lang_choice) - 1]

    # Model
    print(f"\n{Fore.YELLOW}Select model:{Style.RESET_ALL}")
    for i, model in enumerate(MODELS, 1):
        print(f"{i}. {model}")
    model_choice = input(f"{Fore.CYAN}Enter your choice (1-{len(MODELS)}, or press Enter to keep current): {Style.RESET_ALL}")
    if model_choice:
        new_model = MODELS[int(model_choice) - 1]
        if new_model != settings['model']:
            config['DEFAULT']['model'] = new_model
            del transcription_tool
            transcription_tool = TranscriptionTool(new_model)

    # Duration
    duration = input(f"{Fore.CYAN}Enter recording duration in seconds (or press Enter to keep current): {Style.RESET_ALL}")
    if duration:
        config['DEFAULT']['duration'] = duration

    # Auto-tag
    auto_tag = input(f"{Fore.CYAN}Enable auto-tagging? (y/n, or press Enter to keep current): {Style.RESET_ALL}").lower()
    if auto_tag in ['y', 'n']:
        config['DEFAULT']['auto_tag'] = str(auto_tag == 'y')

    # Save the updated config
    with open('work_journal_config.ini', 'w') as configfile:
        config.write(configfile)

    # Update the settings
    settings.update(get_settings())

    print(f"\n{Fore.GREEN}Updated Settings:{Style.RESET_ALL}")
    for key, value in settings.items():
        print(f"{Fore.CYAN}{key.capitalize()}: {value}{Style.RESET_ALL}")

if __name__ == "__main__":
    main_menu()