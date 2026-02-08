import sys
import os
from datetime import datetime
from enum import Enum
import traceback

class LogLevel(Enum):
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4

class Logger:
    def __init__(self, name="Logger", level=LogLevel.INFO, console_output=True):
        self.name = name
        self.level = level
        self.console_output = console_output
        self.log_dir = "logs"
        self._ensure_log_dir()
    
    def _ensure_log_dir(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def _get_color_code(self, level):
        """Get ANSI color codes for different log levels"""
        colors = {
            LogLevel.DEBUG: '\033[36m',    # Cyan
            LogLevel.INFO: '\033[32m',     # Green
            LogLevel.WARN: '\033[33m',     # Yellow
            LogLevel.ERROR: '\033[31m',    # Red
        }
        return colors.get(level, '\033[0m')  # Default to reset
    
    def _log(self, level, message):
        if level.value >= self.level.value:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output = f"[{timestamp}] [{level.name}] {self.name}: {message}"
            
            # Write to console if enabled
            if self.console_output:
                color_code = self._get_color_code(level)
                reset_code = '\033[0m'
                colored_output = f"{color_code}{output}{reset_code}"
                
                # Use stderr for ERROR and WARN, stdout for others
                if level in [LogLevel.ERROR, LogLevel.WARN]:
                    print(colored_output, file=sys.stderr)
                else:
                    print(colored_output)
            
            # Write to file
            log_file = os.path.join(self.log_dir, f"{self.name.lower()}.log")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(output + "\n")
    
    def debug(self, message):
        self._log(LogLevel.DEBUG, message)
    
    def info(self, message):
        self._log(LogLevel.INFO, message)
    
    def warn(self, message):
        self._log(LogLevel.WARN, message)
    
    def error(self, message):
        self._log(LogLevel.ERROR, message)

    def exception(self, exc=None):
        """
        Log exception with error message, line number, and file path
        """
        if exc is None:
            exc_type, exc_value, exc_traceback = sys.exc_info()
        else:
            exc_type = type(exc)
            exc_value = exc
            exc_traceback = exc.__traceback__
        
        if exc_traceback is not None:
            # Get the last frame (where the exception occurred)
            tb = traceback.extract_tb(exc_traceback)
            last_frame = tb[-1]
            
            error_message = str(exc_value)
            file_path = last_frame.filename
            line_no = last_frame.lineno
            
            exception_info = f"Exception: {error_message} | File: {file_path} | Line: {line_no}"
            self._log(LogLevel.ERROR, exception_info)
        else:
            self._log(LogLevel.ERROR, "No exception information available")
    
    def set_console_output(self, enabled):
        """Enable or disable console output"""
        self.console_output = enabled
    
    def set_level(self, level):
        """Change the logging level"""
        self.level = level