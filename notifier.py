import smtplib

def send_email(to_email, subject, body):
    from_email = "your_email@gmail.com"
    password = "your_password"
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(from_email, password)
        message = f"Subject: {subject}\n\n{body}"
        server.sendmail(from_email, to_email, message)

# For SMS, use Twilio (optional)
def send_sms(to_number, body):
    from twilio.rest import Client
    account_sid = 'your_sid'
    auth_token = 'your_token'
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=body,
        from_='+1234567890',
        to=to_number
    ) 