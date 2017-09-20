from datetime import datetime

class TagsTrain:

    user_id = 0
    tag_id = 0
    date = datetime(2017, 12, 5)
    userDate = 0
    color = 0
    clicks = 0
    isIT = 0
    isSP = 0
    isGB = 0

    def __init__(self, user_id, tag_id, date, userDate, color, clicks):
        self.user_id = user_id
        self.tag_id = tag_id
        self.date = date
        self.userDate = userDate
        self.color = color
        self.clicks = clicks

    def setCountries(self, countryCode):
        if countryCode == 0:
            self.isIT = 1
        else:
            if countryCode == 1:
                self.isSP = 1
            else:
                if countryCode == 2:
                    self.isGB = 1