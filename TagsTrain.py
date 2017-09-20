from datetime import datetime

class TagsTrain:

    user_id = 0
    tag_id = 0
    date = datetime(2017, 12, 5)
    userDate = 0
    clicks = 0
    isIT = 0
    isSP = 0
    isGB = 0

    isColor0 = 0
    isColor1 = 0
    isColor2 = 0
    isColor3 = 0
    isColor4 = 0
    isColor5 = 0

    def __init__(self, user_id, tag_id, date, userDate, color, clicks):
        self.user_id = user_id
        self.tag_id = tag_id
        self.date = date
        self.userDate = userDate
        self.color = self.setColor(int(color))
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
                else:
                    print "la estaba cagando"

    def setColor(self, color):
        self.isColor0 = 0
        self.isColor1 = 0
        self.isColor2 = 0
        self.isColor3 = 0
        self.isColor4 = 0
        self.isColor5 = 0
        if color == 0:
            self.isColor0 = 1
        elif color == 1:
            self.isColor1 = 1
        elif color == 2:
            self.isColor2 = 1
        elif color == 3:
            self.isColor3 = 1
        elif color == 4:
            self.isColor4 = 1
        elif color == 5:
            self.isColor5 = 1
        else:
            print "Error in color" + str(color)