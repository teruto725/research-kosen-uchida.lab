
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

output_file = open("output2.csv","wt")
for diagram in diagrams:
    for r in diagram.rlist:
        if r.fromclass.get_natural_name() == r.toclass.get_natural_name():
            continue
        tname = None
        if r.type == "--":
            tname = "k"
        elif r.type == "--*":
            tname = "s1"
        elif r.type == "--|>":
            tname = "h1"
        output_file.write(r.fromclass.get_natural_name()+","
                            +r.toclass.get_natural_name()+","
                            +tname+",analyse,"+str(diagram)
                            +",creately\n")

output_file.close()