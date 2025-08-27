from db import init_db, add_person, delete_person_by_id
import sqlite3

init_db()  # Ensure the database and table exist

# Clear all entries from the database
def clear_db():
    conn = sqlite3.connect('data/missing_persons.db')
    c = conn.cursor()
    c.execute('DELETE FROM persons')
    conn.commit()
    conn.close()

clear_db()

add_person(
    name="Dishant Patel",
    age=21,
    last_seen_location="Home",
    date_missing="2024-01-01",
    additional_info="Student, wearing glasses",
    contact_email="dishantptl04@gmail.com",
    image_path="data/dishant.jpg"
)

print("Database cleared and Dishant added for testing!")
print("Now the app should detect Dishant's face in the camera feed.") 
