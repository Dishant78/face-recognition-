from db import init_db, add_person
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
    contact_email="dishantptl04@gmail.com",
    last_seen_location="home",
    date_missing="2025-08-06",
    image_path="data\dishant1.jpg",
    additional_info="gambler"
    )

print("Database cleared and added for testing!")
print("Now the app should detect Missing face in the camera feed.") 