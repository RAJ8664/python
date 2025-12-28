from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv


load_dotenv()

main_llm = ChatOpenAI(model="deepseek/deepseek-r1-0528:free")
model1 = ChatOpenAI(model="deepseek/deepseek-r1-0528:free")
model2 = ChatOpenAI(model="deepseek/deepseek-r1-0528:free")
merge_llm = ChatOpenAI(model="deepseek/deepseek-r1-0528:free")

"""
Consider a scenario 
1. First we will give a topic
2. Then we will ask llm to generate a brief detailed report on the topic.
3. Then we will generate (notes, quiz) from that detailed report.
4. Output both of them to the user.

┌────────────────────┐
│        User        │
│  (Provides Topic)  │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Topic Ingestion   │
│  & Validation      │
└─────────┬──────────┘
          │
          ▼
┌──────────────────────────────────┐
│        LLM – Main Chain          │
│  Generate Detailed Report        │
│  (Structured, Rich Content)      │
└─────────┬────────────────────────┘
          │
          │  Shared Context
          ▼
┌───────────────────────────────────────────────────┐
│               Parallel Processing                 │
│       LLM1                      LLM2              │
│   ┌───────────────────┐     ┌──────────────────┐  │
│   │   Notes Generator │     │   Quiz Generator │  │
│   │   (Summarization, │     │   (MCQs, Short   │  │
│   │    Key Points)    │     │    Answers)      │  │
│   └─────────┬─────────┘     └─────────┬────────┘  │
│             │                         │           │
└─────────────┼─────────────────────────┼───────────┘
              │                         │
              ▼                         ▼
       ┌────────────────┐       ┌────────────────┐
       │     Notes      │       │      Quiz      │
       └────────┬───────┘       └────────┬───────┘
                │                        │
                └─────────────┬──────────┘
                              ▼
                   ┌────────────────────┐
                   │   Output Composer  │
                   │  (Merge + Format)  │
                   └─────────┬──────────┘
                             │
                             ▼
                   ┌────────────────────┐
                   │    Final Output    │
                   │  • Notes           │
                   │  • Quiz            │
                   └────────────────────┘

NOTE : It is not necessary to have differnet llm for each component, we can use the same llm for all components.

"""

parser = StrOutputParser()

report_template = PromptTemplate(
    template="write a detailed report on the topic: {topic}", input_variables=["topic"]
)

notes_template = PromptTemplate(
    template="write a complete notes on the report: \n {report}",
    input_variables=["report"],
)

quiz_template = PromptTemplate(
    template="write a complete quiz on the report: \n {report}",
    input_variables=["report"],
)

merge_template = PromptTemplate(
    template="""
You are an output composer.

Combine the NOTES and QUIZ into a single, well-formatted final output.
Follow this structure strictly:

## Study Notes
<notes here>

## Quiz
<quiz here>

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"],
)

report_chain = report_template | main_llm | parser

parallel_chain = RunnableParallel(
    notes_chain=notes_template | model1 | parser,
    quiz_chain=quiz_template | model2 | parser,
)

merge_chain = merge_template | merge_llm | parser


def pipeline(topic: str):
    report = report_chain.invoke({"topic": topic})
    parallel_chain_output = parallel_chain.invoke({"report": report})
    notes = parallel_chain_output["notes_chain"]
    quiz = parallel_chain_output["quiz_chain"]
    merge = merge_chain.invoke({"notes": notes, "quiz": quiz})
    return merge


print(pipeline("Linux"))


"""
Output Received : 

## 📘 Study Notes

### Complete Notes on "Comprehensive Report: Linux Operating System"  

#### **1. Introduction**  
- **Definition**: Linux is an **open-source, Unix-like OS kernel** created by *Linus Torvalds* (1991).  
- **Licensing**: Distributed under **GNU GPL**, enabling free use, modification, and redistribution.  
- **Scope**: Term "Linux" commonly refers to **distributions (distros)** bundling the kernel with GNU software and tools.  
- **Market Dominance (2024)**:  
  - Servers: **96.3%** of top 1 million servers.  
  - Mobile: **85%** of smartphones (via Android).  
  - Supercomputing, embedded systems, and cloud infrastructure.  

---

#### **2. History and Development**  
- **1991**: Linus Torvalds initiates Linux as a hobby project inspired by MINIX.  
- **GNU Fusion**: Combined with GNU tools (GCC, Bash) to form a complete OS.  
- **Licensing**: GPLv2 adopted (1992), fostering collaboration (>15,000 contributors).  
- **Key Milestones**:  
  - **1996**: Kernel v2.0 adds symmetric multiprocessing.  
  - **2015**: Systemd standardizes service management.  
  - **2022**: Kernel v5.19 supports ARM, RISC-V, and AI accelerators.  

---

#### **3. Technical Architecture**  
- **Design**: **Monolithic kernel** with modular, dynamically loaded components.  
- **Core Components**:  
  - **Kernel vs. User Space**: Hardware access restricted to kernel via system calls.  
  - **Process Management**: Preemptive multitasking, cgroups for resource control.  
  - **File Systems**: Ext4, Btrfs, XFS; supports >30 file systems.  
  - **Networking**: Robust TCP/IP stack, iptables/nftables (firewalls), KVM virtualization.  
  - **Security**: SELinux/AppArmor (access control), privilege separation via Capabilities.  

---

#### **4. Distributions (Distros)**  
- **Purpose**: Distros package the kernel with tailored software stacks.  
- **Major Categories**:  
  | **Type**       | **Examples**               | **Target Audience**               |  
  |----------------|----------------------------|-----------------------------------|  
  | Enterprise     | RHEL, SUSE Linux Enterprise| Servers, mission-critical systems|  
  | Community      | Fedora, openSUSE           | Developers, enthusiasts           |  
  | Desktop       | Ubuntu, Linux Mint          | General users                     |  
  | Lightweight   | Alpine, Lubuntu            | IoT/legacy hardware              |  
  | Specialized   | Kali Linux, Raspbian       | Security/Raspberry Pi apps        |  
- **Package Managers**: APT (Debian/Ubuntu), RPM (RHEL/Fedora), Pacman (Arch).  

---

#### **5. Market Presence**  
- **Servers/Cloud**: **80% cloud market share** (AWS, Azure, Google Cloud).  
- **Mobile**: Android (Linux-based) holds **71% global OS share**.  
- **Supercomputing**: Powers **100% of Top500 supercomputers** (since 2017).  
- **IoT/Embedded**: Routers (OpenWrt), smart TVs (Tizen), vehicles (Tesla).  
- **Desktop**: **~2.8% market share**, growing in developer/enterprise niches (e.g., Dell/Lenovo preloads).  

---

#### **6. Security Advantages**  
- **Open-Source Model**: Rapid patching (e.g., Heartbleed fixed in hours).  
- **Privilege Model**: Strict root/user separation minimizes attack surfaces.  
- **Auditing**: Community-driven code scrutiny; NSA contributes to SELinux.  
- **Customizability**: Minimalist deployments (e.g., Alpine Linux <8MB).  
- **Note**: Misconfigurations/unpatched software remain vulnerabilities.  

---

#### **7. Challenges**  
- **Fragmentation**: Distro incompatibilities (package formats, init systems).  
- **Driver Support**: Limited vendor support for proprietary hardware.  
- **Desktop Adoption**: Complexity perception; lacks tools like Adobe Suite.  
- **Learning Curve**: Reliance on terminal workflows deters casual users.  

---

#### **8. Future Developments**  
- **Kernel**:  
  - **Rust integration** for safer drivers (v6.1+).  
  - Enhanced laptop **energy efficiency**.  
- **Containers**: Kubernetes/Fedora CoreOS for cloud-native infrastructure.  
- **Desktop**: Growth via **SteamOS (gaming)**, Chromebooks, and **Windows Subsystem for Linux (WSL2)**.  
- **Hardware**: ARM/RISC-V expansion for IoT/edge computing.  

---

#### **9. Conclusion**  
- Linux evolved from a 1991 student project into a **global backbone** for servers, cloud, mobile, and supercomputing.  
- Strengths: **Open-source collaboration, adaptability, security, scalability**.  
- Desktop adoption remains limited but improves via usability initiatives (GNOME/KDE, Proton/Steam Deck).  
- Future-proof in emerging fields (AI, edge computing) due to libre model and community-driven innovation.  

**Sources**: Linux Foundation, IDC, Gartner, Statcounter, Top500.org, Kernel.org.  
**Revision**: August 2024.

---

## 📝 Quiz

### **Linux Operating System Quiz**  

**Instructions:**  
- Choose the best answer for each question.  
- Total questions: 10  
- Topics: History, architecture, distributions, market impact, security, and future trends.  

---

#### **Questions:**  
1. **Who created Linux and when was it first released?**  
   A) Richard Stallman (1983)  
   B) Linus Torvalds (1991)  
   C) Eric Raymond (1997)  
   D) Dennis Ritchie (1970)  

2. **What license ensures Linux remains free to use, modify, and redistribute?**  
   A) MIT License  
   B) Apache License 2.0  
   C) GNU General Public License (GPL)  
   D) BSD License  

3. **Linux uses which type of kernel design?**  
   A) Microkernel  
   B) Hybrid kernel  
   C) Monolithic kernel  
   D) Exokernel  

4. **Which Linux distribution is specifically designed for security auditing and penetration testing?**  
   A) Ubuntu  
   B) Fedora  
   C) Alpine Linux  
   D) Kali Linux  

5. **As of 2024, Linux powers what percentage of the world's top 1 million servers?**  
   A) 71%  
   B) 85%  
   C) 96.3%  
   D) 80%  

6. **What tool enabled granular resource control (CPU/memory) for Linux processes?**  
   A) Systemd  
   B) SELinux  
   C) cgroups  
   D) iptables  

7. **Which challenge does Linux face regarding software ecosystems?**  
   A) Excessive licensing costs  
   B) Lack of developer tools  
   C) Fragmentation between distributions  
   D) Limited hardware compatibility  

8. **Android, which holds 71% of the global mobile OS market, is based on what technology?**  
   A) Windows NT kernel  
   B) macOS Darwin  
   C) Linux kernel  
   D) Unix System V  

9. **What future Linux innovation focuses on memory safety for kernel drivers?**  
   A) Python integration  
   B) Rust integration  
   C) Go integration  
   D) JavaScript bindings  

10. **Linux's "Root/user separation" model primarily enhances what?**  
    A) Energy efficiency  
    B) Network speed  
    C) Security  
    D) Graphical performance  

---

#### **Answer Key:**  
1. **B) Linus Torvalds (1991)**  
   *(Section 2: History and Development)*  
2. **C) GNU General Public License (GPL)**  
   *(Section 1: Introduction)*  
3. **C) Monolithic kernel**  
   *(Section 3: Technical Architecture)*  
4. **D) Kali Linux**  
   *(Section 4: Distributions)*  
5. **C) 96.3%**  
   *(Section 1: Introduction)*  
6. **C) cgroups**  
   *(Section 3: Process Management)*  
7. **C) Fragmentation between distributions**  
   *(Section 7: Challenges)*  
8. **C) Linux kernel**  
   *(Section 5: Linux in the Market)*  
9. **B) Rust integration**  
   *(Section 8: Future Developments)*  
10. **C) Security**  
    *(Section 6: Security Advantages)*  

---  
**Scoring:**  
- 9-10 correct: Linux Expert!  
- 6-8 correct: Knowledgeable User  
- <6 correct: Review the report and retry!  

*Source: Comprehensive Report: Linux Operating System (Revision: August 2024)*





"""
