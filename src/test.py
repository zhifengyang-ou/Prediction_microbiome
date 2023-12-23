file_path = 'config.ini'
desired_encoding = 'utf-8'

with open(file_path, 'rb') as file:
    for line_number, raw_line in enumerate(file, start=1):
        try:
            line = raw_line.decode(desired_encoding)
        except UnicodeDecodeError:
            print(f"Encoding error on line {line_number}")
        else:
            # Process the line as needed
            if 'your_pattern' in line:
                print(f"Line {line_number}: {line.strip()}")
