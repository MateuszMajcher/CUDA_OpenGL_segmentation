#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/glut.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include "helper_math.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <iostream>

int width = 800, height = 600;
float vboWindowScale = 0.5f;
int widthScaled, heightScaled;
double fovy = 60.0, aspect, zNear = 1.0, zFar = 1024.0;
unsigned int FPS = 60, msecs = 1000 / FPS;
double eyeX, eyeY, eyeZ;
bool animation = true;

GLfloat xRotated, yRotated, zRotated;
GLuint R, G, B = 0;

//ilosc kolorow
const int csize = 7;
//kolory dla obiektow
int4 colors[csize];

unsigned int vertVBO = 0, normalVBO = 0;
struct cudaGraphicsResource *cudaVertVBO = NULL, *cudaNormalVBO = NULL;
int vertSize, normalSize;

unsigned int pbo = 0;
struct cudaGraphicsResource *cudaPBO = NULL;
int pboSize;

void initialize();
void resetCamera();
void createVBO();
void deleteVBO();
void recreateVBO();
void createPBO();
void deletePBO();
void recreatePBO();
int exitHandler();
void display();
void reshape(int w, int h);
void keyboard(unsigned char key, int x, int y);
void special(int key, int x, int y);
void timer(int value);
void displayRGB();
void drawGlutObject(int id, GLdouble size, GLfloat X, GLfloat Y, GLfloat Z, int4 color);
void drawTeaPot(int size, GLfloat X, GLfloat Y, GLfloat Z);
void drawSphere(GLdouble size, GLfloat X, GLfloat Y, GLfloat Z);
void drawCube(GLdouble size, GLfloat X, GLfloat Y, GLfloat Z);
void drawTetrahedron(GLdouble size, GLfloat X, GLfloat Y, GLfloat Z);
void randCol(int4[], int size);
 
__device__ __forceinline__ int segmentation(int value, int prog)
{
	return (value < prog) ? 0 : value;
}


__global__ void fancyKernel(uchar3 *pixels,int width, int height, int R,  int G, int B)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;


	if ((x < width) && (y < height))
	{
		int i = y*width + x;

		
		pixels[i].x = segmentation(pixels[i].x, R); // R
		pixels[i].y = segmentation(pixels[i].y, G); // G
		pixels[i].z = segmentation(pixels[i].z, B); // B
		
	}
}

int main(int argc, char *argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutCreateWindow(argv[0]);
	initialize();
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);
	glutTimerFunc(msecs, timer, 0);
	glutMainLoop();
	return 0;
}

void initialize()
{
	//kolory
	GLfloat mat_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat mat_shininess[] = { 50.0f };
	GLfloat light_position[] = { 1.0f, 1.0f, 1.0f, 0.0f };
	randCol(colors, csize);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_LINE_SMOOTH);
	glPointSize(2.0f);
	glLineWidth(2.0f);


	glShadeModel(GL_SMOOTH);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	resetCamera();

	if (cudaSetDevice(0) != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exitHandler();
		exit(EXIT_FAILURE);
	}

	glewExperimental = true;
	if (glewInit() != GLEW_OK)
	{
		fprintf(stderr, "GLEW initialization failed!");
		exitHandler();
		exit(EXIT_FAILURE);
	}

	/*if (glewIsSupported("GL_VERSION_2_0") == false)
	{
	fprintf(stderr, "Extensions are not supported!");
	exitHandler();
	exit(EXIT_FAILURE);
	}*/

	//utworzenie buforów wierzcho³ków
	createVBO();
	//utworzenie Pixel Buffer Object
	createPBO();
}

void resetCamera()
{
	eyeX = 2.0;
	eyeY = 2.0;
	eyeZ = 2.0;
}



void createVBO()
{
	widthScaled = int(width*vboWindowScale);
	heightScaled = int(height*vboWindowScale);



	vertSize = 4 * widthScaled*heightScaled;

	//utworzenie identyfikatora obiektu buforowego
	glGenBuffers(1, &vertVBO);
	//dowiazanie identyfikatora do obiektu buforowego
	//GL_ARRAY_BUFFER - obiekt buforowy tablic wierzcholkow
	glBindBuffer(GL_ARRAY_BUFFER, vertVBO);
	//ladowanie danych do obiektu buforowego
	//GL_DYNAMIC_DRAW - wielokrotne pobieranie danych i wielokrotne ich wykorzystanie do zapisu do obiektu OpenGL,
	glBufferData(GL_ARRAY_BUFFER, vertSize*sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//rejestacja bufora OpenGL
	cudaGraphicsGLRegisterBuffer(&cudaVertVBO, vertVBO, cudaGraphicsRegisterFlagsNone);


	normalSize = 3 * widthScaled*heightScaled;
	glGenBuffers(1, &normalVBO);
	glBindBuffer(GL_ARRAY_BUFFER, normalVBO);
	glBufferData(GL_ARRAY_BUFFER, normalSize*sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGraphicsGLRegisterBuffer(&cudaNormalVBO, normalVBO, cudaGraphicsMapFlagsWriteDiscard);
}

//ususuwanie obiektow VBO
void deleteVBO()
{
	cudaGraphicsUnregisterResource(cudaVertVBO);
	cudaVertVBO = NULL;
	glDeleteBuffers(1, &vertVBO);
	vertVBO = 0;

	cudaGraphicsUnregisterResource(cudaNormalVBO);
	cudaNormalVBO = NULL;
	glDeleteBuffers(1, &normalVBO);
	normalVBO = 0;
}


void recreateVBO()
{
	deleteVBO();
	createVBO();
}


// Utworzenie Pixel Buffer Object
// Rodzaj buforu OpenGl, s³uzacy do przechowywania pikseli
void createPBO()
{
	pboSize = 3 * width*height;
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, pboSize*sizeof(char), 0, GL_DYNAMIC_COPY);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	cudaGraphicsGLRegisterBuffer(&cudaPBO, pbo, cudaGraphicsRegisterFlagsNone);
}
void deletePBO()
{
	cudaGraphicsUnregisterResource(cudaPBO);
	cudaPBO = NULL;
	glDeleteBuffers(1, &pbo);
	pbo = 0;
}
void recreatePBO()
{
	deletePBO();
	createPBO();
}

int exitHandler()
{
	deleteVBO();
	deletePBO();
	if (cudaDeviceReset() != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

void display()
{
	cudaError_t err = cudaSuccess;
	uchar3 *pixels = NULL;

	size_t num_bytes;
	dim3 block_dim(16, 16);
	dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);
	dim3 grid_dim_scaled((widthScaled + block_dim.x - 1) / block_dim.x, (heightScaled + block_dim.y - 1) / block_dim.y);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	//define a viewing transformation
	gluLookAt(eyeX, eyeY, eyeZ,
		0.0, 0.0, 0.0,
		0.0, 1.0, 0.0);

	glMatrixMode(GL_MODELVIEW);
	// czysczenie bufora rysowania
	glClear(GL_COLOR_BUFFER_BIT);
	////
	glLoadIdentity();

	
	drawGlutObject(1, 0.5, 0.0, 0.0, -3.5, colors[0]);
	drawGlutObject(2, 0.5, 1.0, 0.0, -3.5, colors[1]);
	drawGlutObject(3, 0.5, -1.0, 1.0, -2.5, colors[2]);
	drawGlutObject(4, 0.5, 1.0, 1.0, -4.5, colors[3]);
	drawGlutObject(5, 0.5, -1.0, -1.0, -4.5, colors[4]);
	drawGlutObject(6, 0.5, -1.0, 2.0, -3.5, colors[5]);
	drawGlutObject(7, 0.5, 0.5, 1.0, -7.5, colors[6]);
	drawGlutObject(8, -0.5, 0.5, -1.0, -7.5, colors[6]);
	// PBO.
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	cudaGraphicsMapResources(1, &cudaPBO, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&pixels, &num_bytes, cudaPBO);

	fancyKernel << <grid_dim, block_dim >> >(pixels, width, height, R, G, B);
	err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "fancyKernel kernel launch failed: %s\n", cudaGetErrorString(err));
		exitHandler();
		exit(EXIT_FAILURE);
	}
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching fancyKernel kernel!\n", err);
		exitHandler();
		exit(EXIT_FAILURE);
	}

	cudaGraphicsUnmapResources(1, &cudaPBO, 0);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);

	

	glutSwapBuffers();
}

void reshape(int w, int h)
{
	width = (w > 0) ? w : 1;
	height = (h > 0) ? h : 1;
	aspect = (double)width / (double)height;

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fovy, aspect, zNear, zFar);
	glMatrixMode(GL_MODELVIEW);
	recreateVBO();
	recreatePBO();
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case  'q': (R < 255) ? R += 1 : 255; break;
	case  'a': (R > 0) ? R -= 1 : 0; break;
	case  'w': (G < 255) ? G += 1 : 255; break;
	case  's': (G > 0) ? G -= 1 : 0;  break;
	case  'e': (B < 255) ? B += 1 : 255; break;
	case  'd': (B > 0) ? B -= 1 : 0;  break;
	case 'r':
	case 'R': resetCamera(); break;
	case 32:
	{
			   if (animation = !animation)
			   {
				   glutTimerFunc(msecs, timer, 0);
			   }
			   break;
	}
	case 27: exit(exitHandler()); break;
	default:;
	}
	glutPostRedisplay();

	/*wyswietlanie wartosci RGB*/
	system("cls");
	std::cout << "R: " << R << std::endl << "G: " << G << std::endl << "B: " << B << std::endl;

}

void special(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_LEFT: eyeX -= 0.5; break;
	case GLUT_KEY_RIGHT: eyeX += 0.5; break;
	case GLUT_KEY_UP: eyeY += 0.5; break;
	case GLUT_KEY_DOWN: eyeY -= 0.5; break;
	case GLUT_KEY_HOME: eyeZ -= 0.5; break;
	case GLUT_KEY_END: eyeZ += 0.5; break;
	default:;
	}


	glutPostRedisplay();
}

void timer(int value)
{
	if (animation)
	{
		glutPostRedisplay();
		glutTimerFunc(msecs, timer, 0);
	}
}


void randCol(int4 color[], int size) {
	srand(time(NULL));
	for (int i = 0; i < size; i++) {
		int temp = rand() % 255;
		color[i] = { rand() % 255, rand() % 255, rand() % 255, rand() % 255 };
	}
}

void drawTeaPot(int size, GLfloat X, GLfloat Y, GLfloat Z) {
	glPushMatrix();
	glTranslatef(X, Y, Z);
	glRotatef(90, 0.1, 0.2, 0.5);
	glColor3ub(0, 255, 0);
	glutSolidTeapot(size);
	glPopMatrix();
}


void drawGlutObject(int id, GLdouble size, GLfloat X, GLfloat Y, GLfloat Z, int4 color) {
	glPushMatrix();
	glTranslatef(X, Y, Z);
	glRotatef(90, 0.1, 0.2, 0.5);
	glColor3ub(color.x, color.y, color.z);

	switch (id)
	{
	case 1: glutSolidTeapot(size); break;
	case 2: glutSolidSphere(size, 50, 50); break;
	case 3: glutSolidCube(size); break;
	case 4: glutSolidTetrahedron(); break;
	case 5: glutSolidIcosahedron(); break;
	case 6: glutSolidOctahedron(); break;
	case 7: glutSolidDodecahedron(); break;
	case 8:	glutSolidTorus(size, 10, 1, 1);
	default:;
	}

	glPopMatrix();


}



void drawSphere(GLdouble size, GLfloat X, GLfloat Y, GLfloat Z) {
	glPushMatrix();
	glTranslatef(X, Y, Z);
	glRotatef(90, 0.1, 0.2, 0.5);
	glColor3ub(0, 255, 0);
	glutSolidSphere(size, 50, 50);
	glPopMatrix();
}

void drawCube(GLdouble size, GLfloat X, GLfloat Y, GLfloat Z) {
	glPushMatrix();//
	glTranslatef(X, Y, Z);
	glRotatef(90, 0.1, 0.2, 0.5);
	glColor3ub(0, 255, 0);
	glutSolidCube(size);
	glPopMatrix();
}

void drawTetrahedron(GLdouble size, GLfloat X, GLfloat Y, GLfloat Z) {
	glPushMatrix();
	glTranslatef(X, Y, Z);
	glRotatef(90, 0.1, 0.2, 0.5);
	glColor3ub(0, 255, 0);
	glutSolidTetrahedron();
	glPopMatrix();
}