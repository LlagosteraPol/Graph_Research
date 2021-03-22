import React from 'react';

import Layout from '../components/Layout';

import Scroll from '../components/Scroll';

import pic1 from '../assets/images/volunteer_6_1920.jpg';
import pic2 from '../assets/images/projects_6_1280.png';
import pic3 from '../assets/images/associations_2_1920.jpg';
import config from '../../config';
const IndexPage = () => (
  <Layout>
    <section id="banner">
      <div className="inner">
        <h2>{config.heading}</h2>
        <p>{config.subHeading}</p>
        <ul className="actions special">
          <li>
            <Scroll type="id" element="one">
              <a href="/#" className="button primary">
                Explorar
              </a>
            </Scroll>
          </li>
        </ul>
      </div>
      <Scroll type="id" element="one">
        <a href="#one" className="more">
          Saber más
        </a>
      </Scroll>
    </section>

    <section id="one" className="wrapper style1 special">
      <div className="inner">
        <header className="major">
          <h2>
            Un puente entre hacedores y entidades 
          </h2>
          <p>
            Una plataforma digital que conecta a todos aquellos que abogan 
            por causas nobles con las entidades sin ánimo de lucro que 
            las apoyan
          </p>
        </header>
        <ul className="icons major">
          <li>
            <p>Voluntariado</p>
            <Scroll type="id" element="volunteering">
              <a href="#volunteering" className="icon fa-heart major style1"></a>
            </Scroll>
          </li>
          <li>
          <p>Proyectos</p>
            <Scroll type="id" element="projects">
              <a href="#projects" className="icon fa-compass major style2"></a>
            </Scroll>
          </li>
          <li>
          <p>Asociaciones</p>
            <Scroll type="id" element="associations">
              <a href="#associations" className="icon fa-building major style4"></a>
            </Scroll>
          </li>
        </ul>
      </div>
    </section>

    <section id="two" className="wrapper alt style2">
      <section id="volunteering" className="spotlight">
        <div className="image">
          <img src={pic1} alt="" />
        </div>
        <div className="content">
          <h2>
            Voluntariado
          </h2>
          <p>
            Aliquam ut ex ut augue consectetur interdum. Donec hendrerit
            imperdiet. Mauris eleifend fringilla nullam aenean mi ligula.
          </p>
        </div>
      </section>
      <section id="projects" className="spotlight">
        <div className="image">
          <img src={pic2} alt="" />
        </div>
        <div className="content">
          <h2>
            Proyectos
          </h2>
          <p>
            Aliquam ut ex ut augue consectetur interdum. Donec hendrerit
            imperdiet. Mauris eleifend fringilla nullam aenean mi ligula.
          </p>
        </div>
      </section>
      <section id="associations" className="spotlight">
        <div className="image">
          <img src={pic3} alt="" />
        </div>
        <div className="content">
          <h2>
            Asociaciones
          </h2>
          <p>
            Aliquam ut ex ut augue consectetur interdum. Donec hendrerit
            imperdiet. Mauris eleifend fringilla nullam aenean mi ligula.
          </p>
        </div>
      </section>
    </section>

    <section id="three" className="wrapper style6 special">
      <div className="inner">
        <header className="major">
          <h2>Nuestra Plataforma</h2>
          <p>
            Aliquam ut ex ut augue consectetur interdum. Donec amet imperdiet
            eleifend
            <br />
            fringilla tincidunt. Nullam dui leo Aenean mi ligula, rhoncus
            ullamcorper.
          </p>
        </header>
        <ul className="features">
          <li className="icon fa-paper-plane">
            <h3>Todo en uno</h3>
            <p>
              Augue consectetur sed interdum imperdiet et ipsum. Mauris lorem
              tincidunt nullam amet leo Aenean ligula consequat consequat.
            </p>
          </li>
          <li className="icon solid fa-laptop">
            <h3>Mapa</h3>
            <p>
              Augue consectetur sed interdum imperdiet et ipsum. Mauris lorem
              tincidunt nullam amet leo Aenean ligula consequat consequat.
            </p>
          </li>
          <li className="icon solid fa-code">
            <h3>Connexión</h3>
            <p>
              Augue consectetur sed interdum imperdiet et ipsum. Mauris lorem
              tincidunt nullam amet leo Aenean ligula consequat consequat.
            </p>
          </li>
          <li className="icon solid fa-headphones-alt">
            <h3>Tu opinión cuenta</h3>
            <p>
              Augue consectetur sed interdum imperdiet et ipsum. Mauris lorem
              tincidunt nullam amet leo Aenean ligula consequat consequat.
            </p>
          </li>
          <li className="icon fa-heart">
            <h3>Visibilidad</h3>
            <p>
              Augue consectetur sed interdum imperdiet et ipsum. Mauris lorem
              tincidunt nullam amet leo Aenean ligula consequat consequat.
            </p>
          </li>
          <li className="icon fa-flag">
            <h3>Y... ¿que pasa despúes?</h3>
            <p>
              Augue consectetur sed interdum imperdiet et ipsum. Mauris lorem
              tincidunt nullam amet leo Aenean ligula consequat consequat.
            </p>
          </li>
        </ul>
      </div>
    </section>

    <section id="cta" className="wrapper style4">
      <div className="inner">
        <header>
          <h2>Te avisamos cuando todo esté listo</h2>
          <p>
            Aliquam ut ex ut augue consectetur interdum endrerit imperdiet amet
            eleifend fringilla.
          </p>
        </header>
        <ul className="actions stacked">
          <li>
            <a href="/#" className="button fit primary">
              Activate
            </a>
          </li>
          <li>
            <a href="/#" className="button fit">
              Learn More
            </a>
          </li>
        </ul>
      </div>
      <div className="inner">
					<form method="post" action="#">
						<div className="row gtr-uniform">
							<div className="col-6 col-12-xsmall">
								<input type="text" name="demo-name" id="demo-name" defaultValue="" placeholder="Nombre" />
							</div>
							<div className="col-6 col-12-xsmall">
								<input type="email" name="demo-email" id="demo-email" defaultValue="" placeholder="Email" />
							</div>
						</div>
            <div className="col-12">
								<ul className="actions">
									<li><input type="submit" value="Enviar" className="primary" /></li>
								</ul>
						</div>
					</form>
        </div>
    </section>
  </Layout>
);

export default IndexPage;
