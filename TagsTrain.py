from datetime import datetime

class TagsTrain:

    tag_id = 0
    date = datetime(2017, 12, 5)
    color = 0
    clicks = 0

    def __init__(self, tag_id, date, color, clicks):
        self.tag_id = tag_id
        self.date = date
        self.color = color
        self.clicks = clicks