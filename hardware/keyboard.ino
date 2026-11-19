const int numRows = 5;
const int numCols = 11;

// Pin definitions
const int rowPins[numRows] = {A1, A2, A3, A4, A5}; // Row pins (connect to Diode Anodes)
const int colPins[numCols] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // Column pins (connect to Switch Cathodes)

// Keyboard Layout
const String keys[numRows][numCols] = {
  {"ESC", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"},
  {"TAB", "q", "w", "e", "r", "t", "z", "u", "i", "o", "p"},
  {"CAPSLOCK", "a", "s", "d", "f", "g", "h", "j", "k", "l", "Backspace"},
  {"SHIFT", "y", "x", "c", "v", "b", "n", "m", "-", "up", "ENTER"},
  {"STRG", "META", "ALT", "SPACE", "SPACE", "ALTGR", ".", ",", "left", "down", "right"}
};

void setup() {
  Serial.begin(9600);
  
  // Set Column pins as inputs with internal pull-ups
  for (int j = 0; j < numCols; j++) {
    pinMode(colPins[j], INPUT_PULLUP);
  }
  
  // Set Row pins as outputs and set them HIGH (inactive)
  for (int i = 0; i < numRows; i++) {
    pinMode(rowPins[i], OUTPUT);
    digitalWrite(rowPins[i], HIGH);
  }
}

void loop() {
  // Scan each row by pulling it LOW
  for (int row = 0; row < numRows; row++) {
    digitalWrite(rowPins[row], LOW);
    
    // Check each column pin
    for (int col = 0; col < numCols; col++) {
      if (digitalRead(colPins[col]) == LOW) {
        //Serial.print("collums:");
        //Serial.print(col + 1);
        
        Serial.println(keys[numRows - (row+1)][col]);
        
        delay(200); // Debounce delay
      }
    }
    
    // Return row pin to HIGH
    digitalWrite(rowPins[row], HIGH);
  }
}
