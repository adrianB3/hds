# Hide_and_seek_ros

### Hello team, don't forget to follow the tutorials before starting development.

* **Idea** - The hider maps the terrain and finds the best place to hide and the seeker tries to find the hider in the shortest time possible.

* **Idea** - Expand hide and seek game to a _visibility based pursuit and evasion_ in which there are two robots - one is the pursuer and one the evader -> the purpose is that the pursuer tries to find the evader before the evader finds the exit from an enviroment (maze).

#### Basic functionalities:

* Navigation(Mech) and Enviroment Mapping(LIDAR)
* Object Detection and Recognition(Camera)
* Finding efficient paths through the enviroment(Algo)

##### Abbreviations
* Image Processing (Object detection and recognition) --> (IP)
* Algorithms (finding efficient paths through the enviroment, ML) --> (Algo)
* Integration (Making all the modules work together) --> (Int)
* Navigation and Enviroment mapping --> (Nav)
* Documentation and Presentatios --> (Docs)

#### Team Members
* Alina Bacalete @alinabacalete24 -> IP, Algo, Int, Nav
* Ligia Crista @Ligia18 -> IP
* Krisztina Kis @KKrisztina -> Int, Docs
* Adrian Balanescu @adrianB3 -> IP, Algo, Int, Nav, Docs
* Andrei Maciuca @Andrei98 -> IP, Algo

ActionLib tutorial http://library.isr.ist.utl.pt/docs/roswiki/actionlib_tutorials(2f)Tutorials(2f)Writing(20)a(20)Simple(20)Action(20)Server(20)using(20)the(20)Execute(20)Callback(2028)Python(29).html

Path planner requiadlks;jfalkdshgfrments:
 - citire harta si convertire in coordonate fata de pozitia robotului
 - calculare traseu robot astfel:
    - output: x, y (distante(metri)), yaw(grade) -> creare tip nou de mesaj ros
    - mai multe tipuri de pathuri 
                            - parcurgere completa harta
                            - recalculare dinamica a traseului in functie de pozitia obiectului ce trebuie gasit
                            - unknown space exploration
    - pathurile trebuie sa fie diferite pentru hider si seeker
