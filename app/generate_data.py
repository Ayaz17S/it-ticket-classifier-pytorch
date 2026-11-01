"""Generate a synthetic IT-helpdesk ticket dataset.

Real helpdesk tickets are confidential, so we generate realistic ones from
templates. To keep the task honest (not 100% accuracy) the generator adds:
- many phrasings, synonyms and filler words per category
- typos and lower/upper-case noise
- words shared across categories ("laptop", "access", "not working")
- 3% label noise, like real tickets that agents mis-categorise

Run:  python -m app.generate_data
"""
import random
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "tickets.csv"
N_PER_CLASS = 600
LABEL_NOISE = 0.03

TEMPLATES = {
    "network": [
        "{wifi} keeps disconnecting {where}",
        "cannot connect to {vpn} from {where}",
        "internet is very slow {where}, pages not loading",
        "{vpn} connection drops every few minutes",
        "no network access on my {device} after the update",
        "unable to reach the intranet site, {vpn} says timeout",
        "ethernet port at my desk is not working",
        "{wifi} shows connected but no internet",
    ],
    "hardware": [
        "my {device} screen is flickering",
        "{device} battery drains very fast and it gets hot",
        "keyboard keys are not working on my {device}",
        "{device} does not turn on after it fell",
        "need a new mouse and headset, current one is broken",
        "external monitor not detected by my {device}",
        "{device} fan is very loud and the machine shuts down",
        "printer on {floor} is jammed and showing an error",
    ],
    "access": [
        "need access to the {app} shared folder for my project",
        "please give me permission to {app}",
        "my manager approved {app} access but it is still not working",
        "request admin rights to install {tool}",
        "unable to open {app}, it says access denied",
        "please add me to the {app} distribution group",
        "need {tool} installed on my {device} for work",
        "role change: need read access to {app} reports",
    ],
    "password": [
        "forgot my password and the account is locked",
        "password expired, cannot log in to {app}",
        "reset my password please, too many wrong attempts",
        "{mfa} code is not coming on my phone",
        "changed my phone, need to re-register {mfa}",
        "account locked after password change on {device}",
        "cannot sign in, it keeps asking for {mfa} again and again",
        "login fails with invalid credentials even with correct password",
    ],
    "email": [
        "{mail} is not syncing new emails on my {device}",
        "mailbox is full, cannot send emails",
        "emails to the client are bouncing back",
        "calendar invites not showing in {mail}",
        "{mail} keeps crashing when I open attachments",
        "shared mailbox for the team is not visible in {mail}",
        "received a suspicious email asking for my bank details",
        "out of office reply is not working in {mail}",
    ],
    "software": [
        "{app} crashes when I export the report",
        "{tool} shows a license expired error",
        "{app} is very slow after the latest update",
        "error code 0x80070005 while updating {tool}",
        "{tool} is not opening, stuck on loading screen",
        "excel macros stopped working after the patch",
        "teams video freezes during meetings on my {device}",
        "{app} gives a blank page in the browser",
    ],
}

SLOTS = {
    "wifi": ["wifi", "wi-fi", "wireless", "office wifi", "guest wifi"],
    "vpn": ["vpn", "the vpn", "global protect vpn", "cisco anyconnect"],
    "where": ["from home", "in the office", "on 3rd floor", "in the meeting room", "since morning", ""],
    "device": ["laptop", "desktop", "dell laptop", "thinkpad", "macbook", "system", "pc"],
    "floor": ["2nd floor", "3rd floor", "the ground floor", "bay 4"],
    "app": ["sharepoint", "jira", "confluence", "sap", "salesforce", "servicenow", "the hr portal", "power bi"],
    "tool": ["visual studio", "python", "docker desktop", "pycharm", "adobe reader", "zoom", "tableau"],
    "mfa": ["mfa", "otp", "authenticator", "2fa"],
    "mail": ["outlook", "email", "mail app", "outlook web"],
}

PREFIXES = ["", "", "hi team, ", "hello, ", "urgent: ", "please help - ", "hi, ", "dear it, "]
SUFFIXES = ["", "", " please help", " asap", ". thanks", ". this is blocking my work", " since yesterday"]


def add_typo(text: str, rng: random.Random) -> str:
    if len(text) < 10 or rng.random() > 0.3:
        return text
    i = rng.randrange(1, len(text) - 1)
    op = rng.choice(["drop", "swap", "double"])
    if op == "drop":
        return text[:i] + text[i + 1:]
    if op == "swap":
        return text[:i - 1] + text[i] + text[i - 1] + text[i + 1:]
    return text[:i] + text[i] + text[i:]


def make_ticket(category: str, rng: random.Random) -> str:
    template = rng.choice(TEMPLATES[category])
    text = template.format(**{k: rng.choice(v) for k, v in SLOTS.items()})
    text = rng.choice(PREFIXES) + text + rng.choice(SUFFIXES)
    text = add_typo(" ".join(text.split()), rng)
    return text.upper() if rng.random() < 0.05 else text


def generate(seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    categories = list(TEMPLATES)
    rows = []
    for cat in categories:
        for _ in range(N_PER_CLASS):
            label = rng.choice(categories) if rng.random() < LABEL_NOISE else cat
            rows.append({"text": make_ticket(cat, rng), "category": label})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} tickets to {OUT}")
    print(df.category.value_counts())
