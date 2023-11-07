using DecribeOfQ.DecribeOfQr;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DecribeOfQ
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            {
                InitializeComponent();
            }
        }
        private void buttonoff()
        {
            btnA.Enabled = false;
            btnB.Enabled = false;
            btnC.Enabled = false;
            btnD.Enabled = false;
            btnE.Enabled = false;

            btnF.Enabled = false;
            btnG.Enabled = false;
            btnH.Enabled = false;
            btnI.Enabled = false;
            btnJ.Enabled = false;

            btnK.Enabled = false;
            btnL.Enabled = false;
            btnM.Enabled = false;
            btnN.Enabled = false;
            btnO.Enabled = false;

            btnP.Enabled = false;
            btnQ.Enabled = false;
            btnR.Enabled = false;
            btnS.Enabled = false;
            btnT.Enabled = false;
        }
        private void buttonon()
        {
            btnA.Enabled = true;
            btnB.Enabled = true;
            btnC.Enabled = true;
            btnD.Enabled = true;
            btnE.Enabled = true;

            btnF.Enabled = true;
            btnG.Enabled = true;
            btnH.Enabled = true;
            btnI.Enabled = true;
            btnJ.Enabled = true;

            btnK.Enabled = true;
            btnL.Enabled = true;
            btnM.Enabled = true;
            btnN.Enabled = true;
            btnO.Enabled = true;

            btnP.Enabled = true;
            btnQ.Enabled = true;
            btnR.Enabled = true;
            btnS.Enabled = true;
            btnT.Enabled = true;
        }
        


        private void BtnClose_Click(object sender, EventArgs e)
        {
            this.Close();
        }
        private enum StateNameEnum // luu state
        {
            A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U,
            V, W, X, Y, Z
        }
        private void BtnStar_Click(object sender, EventArgs e)
        {
            ricTextQ.Clear();
            buttonon();
            DateTime starttime = DateTime.Now;

            ricTextQ.AppendText("\tA    B      C       D       E \n" +
                                "\tF    G      H       I       J \n" +
                                "\tK    L      M       N       O \n" +
                                "\tP    Q      R       S       T \n" +
                                "\n\n\n");

            ricTextQ.AppendText("Click your chose\n");

            btnB.Click += new EventHandler(BtnA_Click);
            btnC.Click += new EventHandler(BtnA_Click);
            btnD.Click += new EventHandler(BtnA_Click);
            btnE.Click += new EventHandler(BtnA_Click);
            btnF.Click += new EventHandler(BtnA_Click);
            btnG.Click += new EventHandler(BtnA_Click);
            btnH.Click += new EventHandler(BtnA_Click);
            btnI.Click += new EventHandler(BtnA_Click);
            btnJ.Click += new EventHandler(BtnA_Click);
            btnK.Click += new EventHandler(BtnA_Click);
            btnL.Click += new EventHandler(BtnA_Click);
            btnM.Click += new EventHandler(BtnA_Click);
            btnN.Click += new EventHandler(BtnA_Click);
            btnO.Click += new EventHandler(BtnA_Click);
            btnP.Click += new EventHandler(BtnA_Click);
            btnQ.Click += new EventHandler(BtnA_Click);
            btnR.Click += new EventHandler(BtnA_Click);
            btnS.Click += new EventHandler(BtnA_Click);
            btnT.Click += new EventHandler(BtnA_Click);

        }

        public void PathFinding(string start)
        {
            QLearning q = new QLearning
            {
                Episodes = 1000,
                Alpha = 0.1,
                Gamma = 0.9,
                MaxExploreStepsWithinOneEpisode = 1000
            };

            QAction fromTo;
            QState state;
            string stateName;
            string stateNameNext;

            // ----------- Begin Insert the path setup here -----------
            /*
A	B	C	D    ||   A   B   C   D   E
E	F	G	H    ||   F   G   H   I   J
I	J	K	L    ||   K   L   M   N   O
M	N	O	P    ||   P   Q   R   S   T
			
0	-1	100	0    ||   0   0   0   0   0
0	-1	-1	0    ||   0  -1  -1   0   0
0	0	0	0    ||   0  -1   0   0   100
0	0	0	0    ||   0   0   0  -1   0

             * */

            #region qua trinh hoc

            // insert the end states here, e.g. goal state
            q.EndStates.Add(StateNameEnum.O.EnumToString());


            // State A           
            stateName = StateNameEnum.A.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action A -> B
            stateNameNext = StateNameEnum.B.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action A -> F
            stateNameNext = StateNameEnum.F.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // State B           
            stateName = StateNameEnum.B.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action B -> A
            stateNameNext = StateNameEnum.A.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action B -> C
            stateNameNext = StateNameEnum.C.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action B -> G
            stateNameNext = StateNameEnum.G.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0));

            // State C           
            stateName = StateNameEnum.C.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action C -> B
            stateNameNext = StateNameEnum.B.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action C -> D
            stateNameNext = StateNameEnum.D.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action C -> H
            stateNameNext = StateNameEnum.H.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0));

            // State D          
            stateName = StateNameEnum.D.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action D -> C
            stateNameNext = StateNameEnum.C.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action D -> I
            stateNameNext = StateNameEnum.I.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action D -> E
            stateNameNext = StateNameEnum.E.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // State E          
            stateName = StateNameEnum.E.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action E -> D
            stateNameNext = StateNameEnum.D.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // action E -> J
            stateNameNext = StateNameEnum.J.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // State F          
            stateName = StateNameEnum.F.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action F -> A
            stateNameNext = StateNameEnum.A.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action F -> K
            stateNameNext = StateNameEnum.K.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action F -> G
            stateNameNext = StateNameEnum.G.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0));

            // State G           
            stateName = StateNameEnum.G.EnumToString();
            q.AddState(state = new QState(stateName, q));

            // State H          
            stateName = StateNameEnum.H.EnumToString();
            q.AddState(state = new QState(stateName, q));

            // State I          
            stateName = StateNameEnum.I.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action I -> D
            stateNameNext = StateNameEnum.D.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action I -> J
            stateNameNext = StateNameEnum.J.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action I -> N
            stateNameNext = StateNameEnum.N.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action I -> H
            stateNameNext = StateNameEnum.H.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0));


            // State J          
            stateName = StateNameEnum.J.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action J -> E
            stateNameNext = StateNameEnum.E.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action J -> I
            stateNameNext = StateNameEnum.I.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action J -> O
            stateNameNext = StateNameEnum.O.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });

            // State K          
            stateName = StateNameEnum.K.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action K -> F
            stateNameNext = StateNameEnum.F.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action K -> P
            stateNameNext = StateNameEnum.P.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action K -> L
            stateNameNext = StateNameEnum.L.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0));

            // State L          
            stateName = StateNameEnum.L.EnumToString();
            q.AddState(state = new QState(stateName, q));

            // State M         
            stateName = StateNameEnum.M.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action M -> R 
            stateNameNext = StateNameEnum.R.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action M -> N
            stateNameNext = StateNameEnum.N.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action M -> H
            stateNameNext = StateNameEnum.H.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0));
            // action M -> L
            stateNameNext = StateNameEnum.L.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0));

            // State N        
            stateName = StateNameEnum.N.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action N -> M
            stateNameNext = StateNameEnum.M.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action N -> I
            stateNameNext = StateNameEnum.I.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action N -> O
            stateNameNext = StateNameEnum.O.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
            // action N -> S
            stateNameNext = StateNameEnum.S.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0));

            // State P          
            stateName = StateNameEnum.P.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action P -> K
            stateNameNext = StateNameEnum.K.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action P -> Q
            stateNameNext = StateNameEnum.Q.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // State Q          
            stateName = StateNameEnum.Q.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action Q -> R
            stateNameNext = StateNameEnum.R.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action Q -> P
            stateNameNext = StateNameEnum.P.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action Q -> L
            stateNameNext = StateNameEnum.L.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0));

            // State R          
            stateName = StateNameEnum.R.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action R -> Q
            stateNameNext = StateNameEnum.Q.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action R -> M
            stateNameNext = StateNameEnum.M.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action R -> S
            stateNameNext = StateNameEnum.S.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0));

            // State S           
            stateName = StateNameEnum.S.EnumToString();
            q.AddState(state = new QState(stateName, q));

            // State T          
            stateName = StateNameEnum.T.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action T -> O
            stateNameNext = StateNameEnum.O.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
            // action T -> S
            stateNameNext = StateNameEnum.S.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0));



            // ----------- End Insert the path setup here -----------

            #endregion

            #region random vật cản
            /*
            // Introduce a single random obstacle
            Random rand = new Random();
            List<QState> availableStates = q.States.Where(s => s.Actions.Count > 0).ToList();

            if (availableStates.Count >= 2)
            {
                // Shuffle the list of available states to randomly select two different states
                rand = new Random();
                availableStates = availableStates.OrderBy(s => rand.Next()).ToList();

                // Create two obstacles in different states
                QState obstacleState1 = availableStates[0];
                QState obstacleState2 = availableStates[1];

                // Print the state names of the obstacles
                Console.WriteLine("Obstacle state 1: " + obstacleState1.StateName);
                Console.WriteLine("Obstacle state 2: " + obstacleState2.StateName);

                // Remove all actions leading out of the obstacle states
                obstacleState1.Actions.Clear();
                obstacleState2.Actions.Clear();

                // Adjust rewards for actions leading to the obstacle states (e.g., make them highly negative)
                foreach (QState otherState in q.States)
                {
                    if (otherState != obstacleState1 && otherState != obstacleState2)
                    {
                        foreach (QAction action in otherState.Actions)
                        {
                            foreach (QActionResult actionResult in action.ActionsResult)
                            {
                                if (actionResult.StateName == obstacleState1.StateName || actionResult.StateName == obstacleState2.StateName)
                                {
                                    // Adjust the reward to discourage choosing this action
                                    actionResult.Reward = -100;
                                }
                            }
                        }
                    }
                }
            }
            */
            #endregion

            q.RunTraining();
            q.PrintQLearningStructure();


            string ketqua = "";
            ricTextQ.AppendText("\tĐiểm kết là 'O'\n\n");
            Queue<string> que = new Queue<string>();
            try
            {



                if (KTVC(start) || start == "")
                {
                    ketqua = start + " là tường hoặc " + start + " không tồn tại";

                }
                else
                {
                    var diemdau = start;
                    ricTextQ.AppendText("\nĐường đi ngắn nhất :");
                    ketqua = start;

                    var tieptheo = "";
                    while (diemdau != StateNameEnum.O.EnumToString())
                    {

                        for (int i = 0; i < q.States.Count; i++)
                        {

                            if (diemdau == q.States[i].StateName)
                            {
                                //tieptheo = q.States[i].Actions[0].ActionName.To;
                                //Console.Write("->{0} ", tieptheo);
                                que.Enqueue(q.States[i].Actions[0].ActionName.To);
                                if (!KTVC(que.Dequeue()))
                                {
                                    tieptheo = q.States[i].Actions[0].ActionName.To;
                                    ketqua += "->" + tieptheo;
                                }
                                else
                                    que.Enqueue(q.States[i].Actions[1].ActionName.To);
                            }
                        }
                        diemdau = tieptheo;


                    }

                }
                

            }
            catch
            {
                ketqua = "điểm nhập vào là tính năng, không tồn tại, không thể đi \n";
            }
            ricTextQ.AppendText(ketqua);
            dericeofstrees(ketqua);
        }

        private bool KTVC(string a)
        {
            List<string> VatCan = new List<string> { "L", "G", "H", "S" };
            for (int i = 0; i < VatCan.Count; i++)
            {
                if (a == VatCan[i])
                {
                    return true;
                }
            }
            return false;
        }
        private void dericeofstrees(string ketqua)
        {
            char[] firstDelimiters = new char[] { '-' };
            string[] firstParts = ketqua.Split(firstDelimiters, StringSplitOptions.RemoveEmptyEntries);

            List<string> result = new List<string>();

            foreach (string part in firstParts)
            {
                char[] secondDelimiters = new char[] { '>' };
                string[] subParts = part.Split(secondDelimiters, StringSplitOptions.RemoveEmptyEntries);

                foreach (string subPart in subParts)
                {
                    string cleanedSubPart = subPart.Trim();
                    if (!string.IsNullOrEmpty(cleanedSubPart))
                    {
                        result.Add(cleanedSubPart);
                    }
                }
            }

            string[] lettersArray = result.ToArray();

            // Hiển thị kết quả
            foreach (string letter in lettersArray)
            {
                //ricTextQ.AppendText(letter);
                foreach (Control control in this.Controls)
                {
                    if (control is Button) // Kiểm tra nếu control là một button
                    {
                        Button button = (Button)control;
                        if (button.Text == letter)
                        {
                            button.BackColor = Color.Green; 
                        }

                    }
                }

            }
        }
        
        private void beginColor()
        {
            btnA.BackColor = Color.White;
            btnB.BackColor = Color.White;
            btnC.BackColor = Color.White;
            btnD.BackColor = Color.White;
            btnE.BackColor = Color.White;

            btnF.BackColor = Color.White;
            btnG.BackColor = Color.IndianRed;
            btnH.BackColor = Color.IndianRed;
            btnI.BackColor = Color.White;
            btnJ.BackColor = Color.White;

            btnK.BackColor = Color.White;
            btnL.BackColor = Color.IndianRed;
            btnM.BackColor = Color.White;
            btnN.BackColor = Color.White;
            btnO.BackColor = Color.Yellow;

            btnP.BackColor = Color.White;
            btnQ.BackColor = Color.White;
            btnR.BackColor = Color.White;
            btnS.BackColor = Color.IndianRed;
            btnT.BackColor = Color.White;
        }
        private void BtnA_Click(object sender, EventArgs e)
        {
            ricTextQ.Clear();
            beginColor();
            ricTextQ.AppendText("\tA     B      C       D       E \n" +
                                "\tF     G      H       I       J \n" +
                                "\tK     L      M       N       O \n" +
                                "\tP     Q      R       S       T \n" +
                                "\n\n\n");

            //giải quết vấn đề ở đây

            ricTextQ.AppendText("Click your chose\n");
           
            Button clickedButton = sender as Button;
            if (clickedButton != null)
            {
                string buttonName = clickedButton.Text;
                ricTextQ.AppendText("Button được click là: " + buttonName + "\n");

                PathFinding(buttonName);
            }
        }

        private void BtnStop_Click(object sender, EventArgs e)
        {
            buttonoff();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            buttonoff();
        }
    }

    namespace DecribeOfQr
    {
        class QLearning // ds trạng thái(state), giới hạn số lần học (MaxQ)
        {
            #region khai báo các biến qlearning
            public List<QState> States { get; private set; } //danh sách các trạng thái
            public Dictionary<string, QState> StateLookup { get; private set; }// từ điển truy cập các trạng thái

            public double Alpha { get; internal set; } // tỷ lệ học
            public double Gamma { get; internal set; } // tỷ lệ chính sát

            public HashSet<string> EndStates { get; private set; } // trạng thái kết thúc
            public int MaxExploreStepsWithinOneEpisode { get; internal set; } //giới hạn số lần học
            public bool ShowWarning { get; internal set; } // show runtime warnings regarding q-learning
            public int Episodes { get; internal set; } // số lượng tập dữ liệu sau quá trình học
            #endregion
            public QLearning()
            {
                States = new List<QState>();
                StateLookup = new Dictionary<string, QState>();
                EndStates = new HashSet<string>();

                // Default when not set
                MaxExploreStepsWithinOneEpisode = 1000;
                Episodes = 1000;
                Alpha = 0.1;
                Gamma = 0.9;
                ShowWarning = true;
            }

            public void AddState(QState state) // hàm thêm trạng thái
            {
                States.Add(state);
            }

            public void RunTraining()
            {
                QMethod.Validate(this);

                /*       
                For each episode: Select random initial state 
                Do while not reach goal state
                    Select one among all possible actions for the current state 
                    Using this possible action, consider to go to the next state 
                    Get maximum Q value of this next state based on all possible actions                
                    Set the next state as the current state
                */

                // For each episode
                var rand = new Random();
                long maxloopEventCount = 0;

                // Train episodes
                for (long i = 0; i < Episodes; i++)
                {
                    long maxloop = 0;
                    // Select random initial state          
                    int stateIndex = rand.Next(States.Count);
                    QState state = States[stateIndex];
                    QAction action = null;
                    do
                    {
                        if (++maxloop > MaxExploreStepsWithinOneEpisode)
                        {
                            if (ShowWarning)
                            {
                                string msg = string.Format(
                                "{0} !! MAXLOOP state: {1} action: {2}, {3} endstate is to difficult to reach?",
                                ++maxloopEventCount, state, action, "maybe your path setup is wrong or the ");
                                QMethod.Log(msg);
                            }

                            break;
                        }

                        // no actions, skip this state
                        if (state.Actions.Count == 0)
                            break;

                        // Selection strategy is random based on probability
                        int index = rand.Next(state.Actions.Count);
                        action = state.Actions[index];

                        // Using this possible action, consider to go to the next state
                        // Pick random Action outcome
                        QActionResult nextStateResult = action.PickActionByProbability();
                        string nextStateName = nextStateResult.StateName;

                        double q = nextStateResult.QEstimated;
                        double r = nextStateResult.Reward;
                        double maxQ = MaxQ(nextStateName);

                        // Q(s,a)= Q(s,a) + alpha * (R(s,a) + gamma * Max(next state, all actions) - Q(s,a))
                        double value = q + Alpha * (r + Gamma * maxQ - q); // q-learning                  
                        nextStateResult.QValue = value; // update

                        // is end state go to next episode
                        if (EndStates.Contains(nextStateResult.StateName))
                            break;

                        // Set the next state as the current state                    
                        state = StateLookup[nextStateResult.StateName];

                    } while (true);
                }
            }


            double MaxQ(string stateName)
            {
                const double defaultValue = 0;

                if (!StateLookup.ContainsKey(stateName))
                    return defaultValue;

                QState state = StateLookup[stateName];
                var actionsFromState = state.Actions;
                double? maxValue = null;
                foreach (var nextState in actionsFromState)
                {
                    foreach (var actionResult in nextState.ActionsResult)
                    {
                        double value = actionResult.QEstimated;
                        if (value > maxValue || !maxValue.HasValue)
                            maxValue = value;
                    }
                }

                // no update
                if (!maxValue.HasValue && ShowWarning)
                    QMethod.Log(string.Format("Warning: No MaxQ value for stateName {0}",
                        stateName));

                return maxValue.HasValue ? maxValue.Value : defaultValue;
            }

            public void PrintQLearningStructure()
            {
                //Console.WriteLine("** Q-Learning structure **");
                foreach (QState state in States)
                {
                    // Console.WriteLine("State {0}", state.StateName);
                    for (int i = 0; i < state.Actions.Count; i++)
                    {
                        for (int j = i + 1; j < state.Actions.Count; j++)
                        {
                            if (state.Actions[i].ActionsResult[0].QValue < state.Actions[j].ActionsResult[0].QValue)
                            {
                                var a = state.Actions[j];
                                state.Actions[j] = state.Actions[i];
                                state.Actions[i] = a;
                            }
                        }
                    }



                    foreach (QAction action in state.Actions)
                    {

                        //Console.WriteLine("  Action " + action.ActionName);
                        //Console.Write(action.GetActionResults());
                    }
                }


            }

            public void ShowPolicy()
            {
                Console.WriteLine("** Show Policy **");
                foreach (QState state in States)
                {
                    double max = Double.MinValue;
                    string actionName = "nothing";
                    foreach (QAction action in state.Actions)
                    {
                        foreach (QActionResult actionResult in action.ActionsResult)
                        {
                            if (actionResult.QEstimated > max)
                            {
                                max = actionResult.QEstimated;
                                actionName = action.ActionName.ToString();
                            }
                        }
                    }

                    Console.WriteLine(string.Format("From state {0} do action {1}, max QEstimated is {2} \n",
                        state.StateName, actionName, max.Pretty()));
                }
            }


        }

        class QState // gồm có tên và hành động {stateName; Action}
        {
            public string StateName { get; private set; } // tên trạng thái
            public List<QAction> Actions { get; private set; }// danh sách các hành động

            public void AddAction(QAction action)
            {
                Actions.Add(action);
            }

            public QState(string stateName, QLearning q)
            {
                q.StateLookup.Add(stateName, this);
                StateName = stateName;
                Actions = new List<QAction>();
            }

            public override string ToString()
            {
                return string.Format("StateName {0}", StateName);
            }
        }

        class QAction
        {
            private static readonly Random Rand = new Random();
            public QActionName ActionName { get; internal set; } // tên action
            public string CurrentState { get; private set; } // trạng thái hiện tại
            public List<QActionResult> ActionsResult { get; private set; } //kết quả của hành động

            public void AddActionResult(QActionResult actionResult)
            {
                ActionsResult.Add(actionResult);
            }

            public string GetActionResults()
            {
                var sb = new StringBuilder();
                foreach (QActionResult actionResult in ActionsResult)
                    sb.AppendLine("     ActionResult " + actionResult);

                return sb.ToString();
            }

            public QAction(string currentState, QActionName actionName = null)
            {
                CurrentState = currentState;
                ActionsResult = new List<QActionResult>();
                ActionName = actionName;
            }

            // The sum of action outcomes must be close to 1
            public void ValidateActionsResultProbability()
            {
                const double epsilon = 0.1;

                if (ActionsResult.Count == 0)
                    throw new ApplicationException(string.Format(
                        "ValidateActionsResultProbability is invalid, no action results:\n {0}",
                        this));

                double sum = ActionsResult.Sum(a => a.Probability);
                if (Math.Abs(1 - sum) > epsilon)
                    throw new ApplicationException(string.Format(
                        "ValidateActionsResultProbability is invalid:\n {0}", this));
            }

            public QActionResult PickActionByProbability()
            {
                double d = Rand.NextDouble();
                double sum = 0;
                foreach (QActionResult actionResult in ActionsResult)
                {
                    sum += actionResult.Probability;
                    if (d <= sum)
                        return actionResult;
                }

                // we might get here if sum probability is below 1.0 e.g. 0.99 
                // and the d random value is 0.999
                if (ActionsResult.Count > 0)
                    return ActionsResult.Last();

                throw new ApplicationException(string.Format("No PickAction result: {0}", this));
            }

            public override string ToString()
            {
                double sum = ActionsResult.Sum(a => a.Probability);
                return string.Format("ActionName {0} probability sum: {1} actionResultCount {2}",
                    ActionName, sum, ActionsResult.Count);
            }
        }

        class QActionResult
        {
            public string StateName { get; internal set; }
            public string PrevStateName { get; internal set; }
            public double QValue { get; internal set; } // Q value is stored here        
            public double Probability { get; internal set; }
            public double Reward { get; internal set; }

            public double QEstimated
            {
                get { return QValue * Probability; }
            }

            public QActionResult(string CurrentState, string stateNameNext = null,
                double probability = 1, double reward = 0)
            {
                PrevStateName = CurrentState;
                StateName = stateNameNext;
                Probability = probability;
                Reward = reward;
            }

            public override string ToString()
            {
                return string.Format("State {0}, Prob. {1}, Reward {2}, PrevState {3}, QE {4}",
                    StateName, Probability.Pretty(), Reward, PrevStateName, QEstimated.Pretty());
            }
        }

        public class QActionName
        {
            public string From { get; private set; }
            public string To { get; private set; }

            public QActionName(string from, string to = null)
            {
                From = from;
                To = to;
            }

            public override string ToString()
            {
                return GetActionName();
            }

            public string GetActionName()
            {
                if (To == null)
                    return From;
                return QMethod.ActionNameFromTo(From, To);
            }
        }

        static class QMethod
        {
            public static void Log(string s)
            {
                //Console.WriteLine(s);
            }

            public static readonly CultureInfo CultureEnUs = new CultureInfo("en-US");

            public static string ToStringEnUs(this double d)
            {
                return d.ToString("G", CultureEnUs);
            }
            public static string Pretty(this double d)
            {
                return ToStringEnUs(Math.Round(d, 2));
            }

            public static string ActionNameFromTo(string a, string b)
            {
                return string.Format("from_{0}_to_{1}", a, b);
            }

            public static string EnumToString<T>(this T type)
            {
                return Enum.GetName(typeof(T), type);
            }

            public static void ValidateRange(double d, string origin = null)
            {
                if (d < 0 || d > 1)
                {
                    string s = origin ?? string.Empty;
                    throw new ApplicationException(string.Format("ValidateRange error: {0} {1}", d, s));
                }
            }

            public static void Validate(QLearning q)
            {
                foreach (var state in q.States)
                {
                    foreach (var action in state.Actions)
                    {
                        action.ValidateActionsResultProbability();
                    }
                }
            }
        }
    }
}
