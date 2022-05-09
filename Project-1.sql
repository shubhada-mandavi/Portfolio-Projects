SELECT *
FROM PortfolioProject..CovidDeaths$
where continent is not null
order by 3,4

SELECT *
FROM PortfolioProject..CovidVaccinations$
where continent is not null
order by 3,4

Select location, date, total_cases, new_cases, total_deaths, population
FROM PortfolioProject..CovidDeaths$
where continent is not null
order by 1,2

--Looking at Total cases Vs Total deaths
--Shows likelihood of dying if you contract Covid in India

Select location, date, total_cases, total_deaths, (total_deaths/total_cases)*100 as death_percentage
FROM PortfolioProject..CovidDeaths$
Where location like 'India'
order by 1,2

--looking at total cases vs population
--shows what percentage of population got infected

Select location, date, total_cases, total_deaths,population, (total_cases/population)*100 as infection_percentage
FROM PortfolioProject..CovidDeaths$
Where location like 'India'

--Looking at countries with highest infection rate 

Select location,population, max(total_cases) as Highest_infection_count, max((total_cases/population)*100) as infection_percentage
FROM PortfolioProject..CovidDeaths$
where continent is not null
group by location,population
order by 4 desc

-- showing countries with highest death count per population

Select location, max(cast(total_deaths as int)) as Highest_death_count
FROM PortfolioProject..CovidDeaths$
where continent is not null 
group by location,population
order by 2 desc

-- LET'S BREAK THINGS DOWN BY CONTINENT
Select location, max(cast(total_deaths as int)) as Total_death_count
FROM PortfolioProject..CovidDeaths$
where continent is null
group by location 
order by 2 desc

--showing continents with highest death count per population
Select continent, max(cast(total_deaths as int)) as Highest_death_count
FROM PortfolioProject..CovidDeaths$
where continent is not null 
group by continent
order by 2 desc

-- Global Numbers
Select date, sum(new_cases) as date_wise_new_cases, sum(cast(new_deaths as int)) as date_wise_deaths, sum(cast(new_deaths as int))/sum(new_cases) as death_percentage
FROM PortfolioProject..CovidDeaths$
where continent is not null
group by date
order by 1,2

Select sum(new_cases) as date_wise_new_cases, sum(cast(new_deaths as int)) as date_wise_deaths, sum(cast(new_deaths as int))/sum(new_cases) as death_percentage
FROM PortfolioProject..CovidDeaths$
where continent is not null
order by 1,2


-- Looking at total population vs vaccinations

Select dea.continent, dea.location, dea.date, dea.population, vacc.new_vaccinations, 
sum(convert(int,vacc.new_vaccinations)) over (partition by dea.location order by dea.location,dea.date) as Rolling_people_vaccinated
From PortfolioProject..CovidDeaths$ as dea 
Join PortfolioProject..CovidVaccinations$ as vacc
on dea.location = vacc.location and dea.date = vacc.date 
where dea.continent is not null
order by 2,3


-- Use a CTE
With PopvsVacc (Continent,location,date,population,new_vaccinations,Rolling_people_vaccinated)
as
(Select dea.continent, dea.location, dea.date, dea.population, vacc.new_vaccinations, 
sum(convert(int,vacc.new_vaccinations)) over (partition by dea.location order by dea.location,dea.date) as Rolling_people_vaccinated
From PortfolioProject..CovidDeaths$ as dea 
Join PortfolioProject..CovidVaccinations$ as vacc
on dea.location = vacc.location and dea.date = vacc.date 
where dea.continent is not null
)
select *, (Rolling_people_vaccinated/population)*100 AS percentage_vaccinations
from PopvsVacc



--TEMP TABLE

Drop table if exists #PercentPopulationVaccinated
CREATE TABLE #PercentPopulationVaccinated 
(Continent nvarchar(255),
location nvarchar(255) ,
date datetime,
population numeric,
new_vaccinations numeric,
Rolling_people_vaccinated numeric)
 
 Insert into #PercentPopulationVaccinated
 Select dea.continent, dea.location, dea.date, dea.population, vacc.new_vaccinations, 
sum(convert(int,vacc.new_vaccinations)) over (partition by dea.location order by dea.location,dea.date) as Rolling_people_vaccinated
From PortfolioProject..CovidDeaths$ as dea 
Join PortfolioProject..CovidVaccinations$ as vacc
on dea.location = vacc.location and dea.date = vacc.date 
where dea.continent is not null

select *, (Rolling_people_vaccinated/population)*100 AS percentage_vaccinations
from #PercentPopulationVaccinated


-- Creating view to store data for later visualizations
Drop view if exists PercentPopulationVaccinated

Create view PercentPopulationVaccinated as (
Select dea.continent, dea.location, dea.date, dea.population, vacc.new_vaccinations, 
sum(convert(int,vacc.new_vaccinations)) over (partition by dea.location order by dea.location,dea.date) as Rolling_people_vaccinated
From PortfolioProject..CovidDeaths$ as dea 
Join PortfolioProject..CovidVaccinations$ as vacc
on dea.location = vacc.location and dea.date = vacc.date 
where dea.continent is not null)

select *
from PercentPopulationVaccinated