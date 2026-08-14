from crewai_tools import SerperDevTool
from crewai_tools import ScrapeWebsiteTool
from crewai import Agent, Task, Crew, LLM

from crew_llm import crew_llm

def setup_web_scraping_agent():
    """Setup the web scraping agent and related components"""
    search_tool = SerperDevTool()  # Tool for performing web searches
    scrape_website = ScrapeWebsiteTool()  # Tool for extracting data from websites

    # Define the web search agent
    web_search_agent = Agent(
        role="Expert Web Search Agent",
        goal="Identify and retrieve relevant web data for user queries",
        backstory="An expert in identifying valuable web sources for the user's needs",
        allow_delegation=False,
        verbose=True,
        llm=crew_llm
    )

    # Define the web scraping agent
    web_scraper_agent = Agent(
        role="Expert Web Scraper Agent",
        goal="Extract and analyze content from specific web pages identified by the search agent",
        backstory="A highly skilled web scraper, capable of analyzing and summarizing website content accurately",
        allow_delegation=False,
        verbose=True,
        llm=crew_llm
    )

    # Define the web search task
    search_task = Task(
        description=(
            "Identify the most relevant web page or article for the topic: '{topic}'. "
            "Use all available tools to search for and provide a link to a web page "
            "that contains valuable information about the topic. Keep your response concise."
        ),
        expected_output=(
            "A concise summary of the most relevant web page or article for '{topic}', "
            "including the link to the source and key points from the content."
        ),
        tools=[search_tool],
        agent=web_search_agent,
    )

    # Define the web scraping task
    scraping_task = Task(
        description=(
            "Extract and analyze data from the given web page or website. Focus on the key sections "
            "that provide insights into the topic: '{topic}'. Use all available tools to retrieve the content, "
            "and summarize the key findings in a concise manner."
        ),
        expected_output=(
            "A detailed summary of the content from the given web page or website, highlighting the key insights "
            "and explaining their relevance to the topic: '{topic}'. Ensure clarity and conciseness."
        ),
        tools=[scrape_website],
        agent=web_scraper_agent,
    )

    # Define the crew to manage agents and tasks
    crew = Crew(
        agents=[web_search_agent, web_scraper_agent],
        tasks=[search_task, scraping_task],
        verbose=1,
        memory=False,
        max_iter=3,
    )
    return crew



def get_web_content(query):
    """Get content from web scraping"""
    crew = setup_web_scraping_agent()
    result = crew.kickoff(inputs={"topic": query})
    # try:
    #     print("Starting crew")
    #     result = crew.kickoff(inputs={"topic": query})
    #     print("Result:", result)
    # except Exception:
    #     time.sleep(5)
    return result.raw
