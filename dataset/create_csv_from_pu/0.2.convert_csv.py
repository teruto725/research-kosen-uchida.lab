#puファイルから語句とwordNetとの対応をcsvファイルで保存する
from model import Diagram,Class,Relation,read_Pu
"""
FILEPATH = "dataset/jisaku/"
filenames = ["ショッピングサイト1.pu",
            "ショッピングサイト2.pu",
            "すごろく1.pu",
            "すごろく2.pu",
            "ホテル予約システム1.pu",
            "ホテル予約システム2.pu",
            "将棋1.pu",
            "将棋2.pu",
            "電気自動車1.pu",
            "電気自動車2.pu"]
"""
FILEPATH = "dataset/creately/"
filenames = [
            "CustomerMerchant.pu",
            "FlightReservationSystem.pu",
            "HospitalManagement.pu",
            "LibraryManagement.pu",
            "monika.pu",
            "OnlineBusReservation.pu",
            "SeminarClassDiagram.pu",
            "SGC.pu",
            "TaskManagement.pu",
            "OrderClassDiagramTemplate.pu"
            ]
diagrams = list()

for file_name in filenames:
    diagram = read_Pu(FILEPATH+file_name)
    diagrams.append(diagram)
    print("ReadIsDone:"+str(diagram))

di ={}
for diagram in diagrams:
    for c in diagram.clist:
        for name in c.namelist:
            di[name.split(".")[0]]=name
            

output_file = open("output4.csv","wt")
for key, value in di.items():
    output_file.write(key+","+value+"\n")
output_file.close()